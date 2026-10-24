import os
from math import sqrt, isqrt

import torch
import numpy as np
from torch.nn import functional as F
from plyfile import PlyData, PlyElement
from einops import einsum
from e3nn.o3 import matrix_to_angles, wigner_D
from pytorch3d.transforms.rotation_conversions import _axis_angle_rotation
from pytorch3d.transforms import (
	random_rotation,
	quaternion_multiply,
	matrix_to_quaternion,
	quaternion_to_matrix,
	quaternion_to_axis_angle,
)

from .utils_sh import SH2RGB


def rotate_shs(gs_attr, rotmat):
	# https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2147223570
	sh = gs_attr['c'] 			# [n, 3, (sh_degree+1)**2]
	P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float() # switch axes: yzx -> xyz
	permuted_rotation_matrix = torch.linalg.inv(P) @ rotmat @ P
	rot_angles = matrix_to_angles(permuted_rotation_matrix)
	
	sh_degree = isqrt(sh.shape[-1])
	# skip rotation for 0-th sh 
	for degree in range(1, sh_degree):
		sh_rotations = wigner_D(degree, rot_angles[0], - rot_angles[1], rot_angles[2]).float()
		sh[..., degree**2 : (degree + 1) ** 2] = einsum(
			sh_rotations,
			sh[..., degree**2 : (degree + 1) ** 2],
			"... i j, ... j -> ... i",
		)
	gs_attr['c'] =sh

	return gs_attr


def rotate_xyz(gs_attr, rotmat, center=None):
	xyz = gs_attr['p']
	if center is None:
		center = torch.mean(xyz, 0)
	xyz = xyz - center
	xyz = xyz @ rotmat.T
	gs_attr['p'] = xyz + center
	if 'pc_xyz' in gs_attr.keys():
		pc_xyz = (gs_attr['pc_xyz'] - center)
		pc_xyz = pc_xyz @ rotmat.T + center
		gs_attr['pc_xyz'] = pc_xyz
	return gs_attr


def rotate_q(gs_attr, rotmat):
	rotation = matrix_to_quaternion(rotmat)
	gs_attr['q'] = quaternion_multiply(rotation, gs_attr['q'])
	return gs_attr


def rotate_gs(gs_attr, rotmat, center=None):
	gs_attr = rotate_xyz(gs_attr, rotmat, center)
	if 'q' in gs_attr.keys():
		gs_attr = rotate_q(gs_attr, rotmat)
	if 'c' in gs_attr.keys():
		gs_attr = rotate_shs(gs_attr, rotmat)
	if 'n' in gs_attr.keys():
		gs_attr['n'] = gs_attr['n'] @ rotmat.T
	return gs_attr


class Rotation():
	"""rotate Gaussian Splats"""
	def __init__(self, rot_type: str='none'):
		assert rot_type in ('x', 'y', 'z', 'so3')
		self.rot_type = rot_type

	def __call__(self, gs_attr, pose=None, angle=None):
		if self.rot_type in ('x', 'y', 'z'):
			if angle is None:
				angle = torch.rand(()) * 2 * torch.pi
			else:
				if not isinstance(angle, torch.Tensor):
					angle = torch.tensor(angle)
			R = _axis_angle_rotation(axis=self.rot_type.upper(), angle=angle)  # (3, 3)
		elif self.rot_type == 'so3':
			R = random_rotation()
		
		gs_attr = rotate_gs(gs_attr, R)
		
		if isinstance(pose, torch.Tensor):
			assert (pose.shape[0] == 4) and (pose.shape[1] == 4), "the dim of pose given is wrong"
			pose[:3, :3] = R @ pose[:3, :3]
			pose[:3, 3] = R @ pose[:3, 3]
			return gs_attr, pose
		else:
			return gs_attr


def load_plydata(path, gs_activate=True, sh_degree=0, input_type='pocsq'):
	plydata = PlyData.read(path)['vertex']

	schema = {
		'p': (['x', 'y', 'z'], None),
		'o': (['opacity'], torch.sigmoid if gs_activate else None),
		's': ([f'scale_{i}' for i in range(3)], torch.exp if gs_activate else None),
		'q': ([f'rot_{i}'   for i in range(4)], lambda x: torch.nn.functional.normalize(x, dim=1)),
		'c': (['f_dc_0', 'f_dc_1', 'f_dc_2'] + \
			  ([f'f_rest_{i}' for i in range(3 * ((sh_degree + 1)**2 - 1))] if sh_degree > 0 else []), 
			  None)
	}

	gs_attr = {}
	for key in input_type:
		props, act_fn = schema.get(key, ([], None))
		valid_props = [p for p in props if p in plydata]
		if len(valid_props) < len(props):
			print(f"Warning: Expected {len(props)} props, found {len(valid_props)}")
			continue
		data = np.stack([plydata[p] for p in valid_props], axis=-1)
		tensor = torch.from_numpy(data).float()
		if act_fn:
			tensor = act_fn(tensor)
		if key != 'c':
			gs_attr[key] = tensor 
		else:
			dc = tensor[:, :3].reshape(-1, 3, 1)
			if sh_degree>0:
				rest = tensor[:, 3:].reshape(-1, 3, (sh_degree + 1)**2 - 1)
				dc = torch.cat([dc, rest], dim=-1)
			gs_attr[key] = dc
	return gs_attr


def save_plydata(attr, path, deactivate=True):
	os.makedirs(os.path.dirname(path), exist_ok=True)

	properties = ['x', 'y', 'z', 
			   	  'nx', 'ny', 'nz',
				  'opacity', 
				  'scale_0', 'scale_1', 'scale_2', 
				  'rot_0', 'rot_1', 'rot_2', 'rot_3', 
				  'f_dc_0', 'f_dc_1', 'f_dc_2']

	for i in range(attr['c'][:, :, 1:].shape[1] * attr['c'][:, :, 1:].shape[2]):
		properties.append('f_rest_{}'.format(i))

	xyz = attr['p'].contiguous().detach().cpu().numpy()
	normals = np.zeros_like(xyz)
	f_dc = attr['c'][:, :, :1].detach().flatten(start_dim=1).contiguous().cpu().numpy()
	f_rest = attr['c'][:, :, 1:].detach().flatten(start_dim=1).contiguous().cpu().numpy()
	if deactivate:
		opacities = torch.log(attr['o']/(1-attr['o'])).detach().cpu().numpy()
		scale = torch.log(attr['s']).detach().cpu().numpy()
	else:
		opacities = attr['o'].detach().cpu().numpy()
		scale = attr['s'].detach().cpu().numpy()
	rotation = attr['q'].detach().cpu().numpy()
	
	dtype_full = [(attribute, 'f4') for attribute in properties]
	
	elements = np.empty(xyz.shape[0], dtype=dtype_full)
	attributes = np.concatenate((xyz, normals, opacities, scale, rotation, f_dc, f_rest), axis=1)
	elements[:] = list(map(tuple, attributes))
	el = PlyElement.describe(elements, 'vertex')
	PlyData([el]).write(path)


def attr_normalize(gs_attr):
	xyz = gs_attr['p']
	centroid = torch.mean(xyz, dim=0)
	xyz = xyz - centroid
	m = torch.amax(torch.linalg.norm(xyz, ord=2, dim=-1)) + 1e-6
	gs_attr['p'] = xyz / m
	if 's' in gs_attr.keys():
		gs_attr['s'] = gs_attr['s'] / m
	if 'pc_xyz' in gs_attr.keys():
		gs_attr['pc_xyz'] = (gs_attr['pc_xyz'] - centroid) / m
	return gs_attr


def center_shift(gs_attr, apply_z=True):
	"""Shift the point cloud like SceneSplat's CenterShift transform."""
	xyz = gs_attr['p']
	xyz_min = xyz.amin(dim=0)
	xyz_max = xyz.amax(dim=0)
	shift = (xyz_min + xyz_max) * 0.5
	shift[2] = xyz_min[2] if apply_z else 0
	gs_attr['p'] = xyz - shift
	if 'pc_xyz' in gs_attr:
		gs_attr['pc_xyz'] = gs_attr['pc_xyz'] - shift
	return gs_attr


def gaussianmae_normalize(gs_attr):
	if 'o' in gs_attr.keys():
		gs_attr['o'] = gs_attr['o'] * 2 - 1
	if 's' in gs_attr.keys():
		scale = gs_attr['s']
		s_center = torch.mean(scale, dim=0)
		scale = scale - s_center
		m = torch.amax(torch.linalg.norm(scale, ord=2, dim=-1)) + 1e-6
		gs_attr['s'] = scale / m
	if 'c' in gs_attr.keys():
		sh = gs_attr['c']
		sh = sh * 0.28209479177387814 
		sh = torch.clip(sh, -0.5, 0.5)
		sh = 2 * sh / sqrt(3)  
		gs_attr['c'] = sh
	return gs_attr


def random_scale(gs_attr, scale_low=0.8, scale_high=1.25):
	scale = torch.empty(1).uniform_(scale_low, scale_high)
	gs_attr['p'] = gs_attr['p'] * scale
	if 's' in gs_attr.keys():
		gs_attr['s'] = gs_attr['s'] * scale
	if 'pc_xyz' in gs_attr.keys():
		gs_attr['pc_xyz'] = gs_attr['pc_xyz'] * scale
	return gs_attr


def random_scale_normal(gs_attr, scale_low=0.8, scale_high=1.25):
	scale = torch.empty(1, 3).uniform_(scale_low, scale_high)
	gs_attr['p'] = gs_attr['p'] * scale
	if 's' in gs_attr.keys():
		gs_attr['s'] = gs_attr['s'] * scale
	if 'pc_xyz' in gs_attr.keys():
		gs_attr['pc_xyz'] = gs_attr['pc_xyz'] * scale
	if 'n' in gs_attr.keys():
		normal = gs_attr['n']
		if torch.rand(1) < 0.2:
			normal.fill_(0.)
		else:
			normal *= scale[:, [1, 2, 0]] * scale[:, [2, 0, 1]]
			normal = F.normalize(normal, p=2, dim=-1, eps=1e-8)
	gs_attr['n'] = normal
	return gs_attr


def random_shift(gs_attr, shift_range=0.1):
	shift = torch.empty(1, 3).uniform_(-shift_range, shift_range)
	gs_attr['p'] = gs_attr['p'] + shift
	if 'pc_xyz' in gs_attr.keys():
		gs_attr['pc_xyz'] = gs_attr['pc_xyz'] + shift
	return gs_attr


def random_jitter(gs_attr, sigma=0.01, clip=0.05):
	noise = torch.clip(torch.randn(*gs_attr['p'].shape) * sigma, -clip, clip)
	gs_attr['p'] = gs_attr['p'] + noise
	return gs_attr


def shuffle(gs_attr):
	perm = torch.randperm(gs_attr['p'].shape[0])
	for key in gs_attr.keys():
		if key in "pocsqn" or key=='point_label':
			gs_attr[key] = gs_attr[key][perm]
	return gs_attr


def random_dropout(gs_attr, max_dropout_ratio=0.5):
	dropout_ratio =  torch.rand(1) * max_dropout_ratio
	drop_idx = torch.where(torch.rand([gs_attr['p'].shape[0],]) <= dropout_ratio)[0]
	if len(drop_idx) > 0:
		for key, value in gs_attr.items():
			if key in "pocsqn" or key=='point_label':
				v = value.clone()
				v[drop_idx] = gs_attr[key][0]
				gs_attr[key] = v
	return gs_attr


def random_rotate(gs_attr):
	R = random_rotation()
	gs_attr = rotate_gs(gs_attr, R)
	return gs_attr


def random_rotate_x(gs_attr):
	angle = torch.rand(()) * 2 * torch.pi
	R = _axis_angle_rotation(axis='X', angle=angle)
	gs_attr = rotate_gs(gs_attr, R)
	return gs_attr


def random_rotate_y(gs_attr):
	angle = torch.rand(()) * 2 * torch.pi
	R = _axis_angle_rotation(axis='Y', angle=angle)
	gs_attr = rotate_gs(gs_attr, R)
	return gs_attr


def random_rotate_z(gs_attr):
	angle = torch.rand(()) * 2 * torch.pi
	R = _axis_angle_rotation(axis='Z', angle=angle)
	gs_attr = rotate_gs(gs_attr, R)
	return gs_attr


def _random_rotate_small(gs_attr, axis, angle=1 / 64, p=0.5):
	"""SceneSplat-style small rotation; angle is measured in multiples of pi."""
	if torch.rand((), device=gs_attr['p'].device) >= p:
		return gs_attr
	angle = torch.empty((), device=gs_attr['p'].device).uniform_(-angle, angle)
	R = _axis_angle_rotation(axis=axis, angle=angle * torch.pi)
	center = (gs_attr['p'].amin(dim=0) + gs_attr['p'].amax(dim=0)) * 0.5
	return rotate_gs(gs_attr, R, center=center)


def random_rotate_x_small(gs_attr, angle=1 / 64, p=0.5):
	return _random_rotate_small(gs_attr, 'X', angle, p)


def random_rotate_y_small(gs_attr, angle=1 / 64, p=0.5):
	return _random_rotate_small(gs_attr, 'Y', angle, p)


def random_flip(gs_attr, p=0.5):
	"""SceneSplat-style independent X/Y reflection around the origin."""
	device = gs_attr['p'].device
	dtype = gs_attr['p'].dtype
	signs = torch.ones(3, device=device, dtype=dtype)
	if torch.rand((), device=device) < p:
		signs[0] = -1
	if torch.rand((), device=device) < p:
		signs[1] = -1

	if torch.all(signs == 1):
		return gs_attr

	gs_attr['p'] = gs_attr['p'] * signs
	if 'pc_xyz' in gs_attr:
		gs_attr['pc_xyz'] = gs_attr['pc_xyz'] * signs
	if 'n' in gs_attr:
		gs_attr['n'] = gs_attr['n'] * signs
	if 'q' in gs_attr:
		R_reflect = torch.diag(signs).to(gs_attr['q'])
		qmat = quaternion_to_matrix(gs_attr['q'])
		qmat = R_reflect @ qmat @ R_reflect
		gs_attr['q'] = matrix_to_quaternion(qmat)
	return gs_attr


def chromatic_auto_contrast(gs_attr, p=0.2, blend_factor=None):
	"""Blend RGB with its per-scene auto-contrast result; RGB is in [0, 1]."""
	if 'c' not in gs_attr or torch.rand((), device=gs_attr['c'].device) >= p:
		return gs_attr
	color = gs_attr['c']
	lo, hi = color.amin(0, keepdim=True), color.amax(0, keepdim=True)
	contrast = (color - lo) / (hi - lo).clamp_min(torch.finfo(color.dtype).eps)
	blend = (torch.rand((), device=color.device, dtype=color.dtype)
			 if blend_factor is None else blend_factor)
	gs_attr['c'] = torch.lerp(color, contrast, blend).clamp_(0, 1)
	return gs_attr


def chromatic_translation(gs_attr, p=0.8, ratio=0.05):
	"""Apply one random RGB offset to a scene; RGB is in [0, 1]."""
	if 'c' in gs_attr and torch.rand((), device=gs_attr['c'].device) < p:
		color = gs_attr['c']
		shift = color.new_empty(1, color.shape[-1]).uniform_(-ratio, ratio)
		gs_attr['c'] = (color + shift).clamp_(0, 1)
	return gs_attr


def chromatic_jitter(gs_attr, p=0.5, std=0.01):
	"""Add independent Gaussian RGB noise; RGB is in [0, 1]."""
	if 'c' in gs_attr and torch.rand((), device=gs_attr['c'].device) < p:
		color = gs_attr['c']
		gs_attr['c'] = (color + torch.randn_like(color) * std).clamp_(0, 1)
	return gs_attr


def sh2rgb(gs_attr):
	if 'c' in gs_attr.keys():
		gs_attr['c'] = SH2RGB(gs_attr['c'].squeeze(-1))
	return gs_attr


def q2angle(gs_attr):
	if 'q' in gs_attr.keys():
		gs_attr['q'] = quaternion_to_axis_angle(gs_attr['q'])
	return gs_attr


TRANSFORMS_MAP = {
	'shuffle': shuffle,
	'attr_normalize': attr_normalize,
	'center_shift': center_shift,
	'random_scale': random_scale,
	'random_scale_normal': random_scale_normal,
	'random_shift': random_shift,
	'random_jitter': random_jitter,
	'random_dropout': random_dropout,
	'random_rotate': random_rotate,
	'random_rotate_x': random_rotate_x,
	'random_rotate_y': random_rotate_y,
	'random_rotate_z': random_rotate_z,
	'random_rotate_x_small': random_rotate_x_small,
	'random_rotate_y_small': random_rotate_y_small,
	'random_flip': random_flip,
	'chromatic_auto_contrast': chromatic_auto_contrast,
	'chromatic_translation': chromatic_translation,
	'chromatic_jitter': chromatic_jitter,
	'sh2rgb': sh2rgb,
	'q2angle': q2angle,
	'gaussianmae_normalize': gaussianmae_normalize,
}


def transform(gs_attr, transform_list):
	for trans_item in transform_list:
		parts = [p.strip() for p in trans_item.split('/')]
		name = parts[0]  # "random_scale"
		args = parts[1:] # ["0.8", "1.2"]
		
		if name in TRANSFORMS_MAP:
			if not args:
				gs_attr = TRANSFORMS_MAP[name](gs_attr)
			else:
				numeric_args = [float(a) for a in args]
				gs_attr = TRANSFORMS_MAP[name](gs_attr, *numeric_args)
		else:
			raise KeyError(f"Transform {name} 不在 TRANSFORMS_MAP 中")

	if 'c' in gs_attr.keys():
		gs_attr['c'] = gs_attr['c'].squeeze(-1)
	return gs_attr
