import math
import random

import torch
import torch.nn as nn
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .utils_sh import *


def create_pose(dist=1.0, theta=0., phi=0.):
    """
    camera pose:
        theta: 控制绕竖直方向旋转 (Yaw)
        phi:   控制绕水平方向旋转 (Pitch)
    """
    theta_rad = torch.tensor(theta) * 2 * torch.pi 
    phi_rad = torch.tensor(phi) * 2 * torch.pi 
    
    # 1. 初始基础位姿：相机正面对着原点，距离为 dist
    # 此时相机的 右=[-1,0,0], 上=[0,1,0], 前=[0,0,-1]
    base_c2w = torch.tensor([
        [-1.,  0.,  0.,   0.],
        [ 0.,  1.,  0.,   0.],
        [ 0.,  0., -1., dist],
        [ 0.,  0.,  0.,   1.]
    ], dtype=torch.float32)
    
    # 2. 绕相机的 X 轴旋转 (Pitch 俯仰角，控制 phi)
    cos_p, sin_p = torch.cos(phi_rad), torch.sin(phi_rad)
    Rx = torch.tensor([
        [1.,    0.,     0., 0.],
        [0., cos_p, -sin_p, 0.],
        [0., sin_p,  cos_p, 0.],
        [0.,    0.,     0., 1.]
    ], dtype=torch.float32)
    
    # 3. 绕世界的 Y 轴旋转 (Yaw 偏航角，控制 theta)
    cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)
    Ry = torch.tensor([
        [ cos_t, 0., sin_t, 0.],
        [    0., 1.,    0., 0.],
        [-sin_t, 0., cos_t, 0.],
        [    0., 0.,    0., 1.]
    ], dtype=torch.float32)
    
    # 4. 组合变换：先执行俯仰旋转(Rx)，再执行水平旋转(Ry)
    c2w = Ry @ Rx @ base_c2w
    
    # 5. 保留 3DGS/Colmap 坐标系的特定反转 (Z轴翻转和X轴翻转)
    trans = torch.eye(4, dtype=torch.float32)
    trans[0, 0] = -1
    trans[2, 2] = -1
    c2w = trans @ c2w
    
    return c2w

def getProjectionMatrix(znear, zfar, fovX, fovY, device):
	"""
	fovX(Y): rad
	return torch.tensor [4, 4]
	"""
	tanHalfFovY = math.tan((fovY / 2))
	tanHalfFovX = math.tan((fovX / 2))
	top = tanHalfFovY * znear
	bottom = -top
	right = tanHalfFovX * znear
	left = -right
	P = torch.zeros(4, 4)
	z_sign = 1.0
	P[0, 0] = 2.0 * znear / (right - left)
	P[1, 1] = 2.0 * znear / (top - bottom)
	P[0, 2] = (right + left) / (right - left)
	P[1, 2] = (top + bottom) / (top - bottom)
	P[3, 2] = z_sign
	P[2, 2] = z_sign * zfar / (zfar - znear)
	P[2, 3] = -(zfar * znear) / (zfar - znear)
	return P.to(device)


class View:
	def __init__(self, radius, FoV_XY, size_HW, scale, shift, device, random_pose=None, zfar=100, znear=0.01):
		# FoV_X/Y: degree
		# random_pose: c2w[Not CUDA MODE]
		# params all in cpu
		self.FoVx, self.FoVy = FoV_XY
		self.FoVx *= (math.pi / 180)
		self.FoVy *= (math.pi / 180)
		self.image_height, self.image_width = size_HW
		self.zfar = zfar
		self.znear = znear
		self.device = device

		# CUDA MODE
		# use camera pose to replace random_c2w_toward_O(c2w)[not CUDA MODE]
		if not isinstance(random_pose, torch.Tensor):
			if isinstance(scale, torch.Tensor):
				scale = scale.cpu()
			if isinstance(shift, torch.Tensor):
				shift = shift.cpu()
			random_pose = create_pose(dist=radius, theta=random.random(), phi=random.random()*0.)
		self.load_c2w(random_pose, scale, shift)

	def generate(self):
		self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, device=self.device).transpose(0,1)
		self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
		self.camera_center = self.world_view_transform.inverse()[3, :3]
	
	def load_c2w(self, pose, scale=1, shift=0):
		# pose: [not CUDA MODE]
		assert isinstance(pose, torch.Tensor)
		assert (pose.shape[0] == 4) and (pose.shape[1] == 4), "the dim of pose given is wrong."
		if isinstance(scale, torch.Tensor):
			scale = scale.cpu()
		if isinstance(shift, torch.Tensor):
			shift = shift.cpu()
		pose = pose.cpu()
		pose[:3, 3] = scale * pose[:3, 3] + shift
		self.world_view_transform = pose.inverse().transpose(0, 1).to(self.device)
		self.generate()

	def get_c2w(self):
		return self.world_view_transform.transpose(0, 1).inverse().cpu()


class GaussianModel(object):
	def __init__(self, sh_degree: int=0, if_activate:bool=False):
		self.sh_degree = sh_degree
		self.if_activate = if_activate
		self._xyz = torch.empty(0)
		self._features_dc = torch.empty(0)
		self._features_rest = torch.empty(0)
		self._scaling = torch.empty(0)
		self._rotation = torch.empty(0)
		self._opacity = torch.empty(0)
		self.scaling_activation = torch.exp
		self.opacity_activation = torch.sigmoid
		self.rotation_activation = nn.functional.normalize
		
	@property
	def get_scaling(self):
		return self.scaling_activation(self._scaling) if self.if_activate else self._scaling
	
	@property
	def get_rotation(self):
		return self.rotation_activation(self._rotation) if self.if_activate else self._rotation
	
	@property
	def get_xyz(self):
		return self._xyz
	
	@property
	def get_features(self):
		return torch.cat((self._features_dc, self._features_rest), dim=1)
	
	@property
	def get_opacity(self):
		return self.opacity_activation(self._opacity) if self.if_activate else self._opacity
	
	def load_attr_torch(self, attr):
		self._xyz = attr['p']
		self._opacity = attr['o']
		self._scaling = attr['s']
		self._rotation = attr['q']
		self._features_dc = attr['c'][:, :, :1].transpose(-1, -2).contiguous()
		self._features_rest = attr['c'][:, :, 1:].transpose(-1, -2).contiguous()

	def render(self, viewpoint_camera: View, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
 
		"""
		Render the scene. 
		Background tensor (bg_color) must be on GPU!
		override_color: None or [N, 3]
		"""
		device = bg_color.device
		# Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
		screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, device=device) + 0

		# Set up rasterization configuration
		tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
		tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

		raster_settings = GaussianRasterizationSettings(
			image_height=int(viewpoint_camera.image_height),
			image_width=int(viewpoint_camera.image_width),
			tanfovx=tanfovx,
			tanfovy=tanfovy,
			bg=bg_color,
			scale_modifier=scaling_modifier,
			viewmatrix=viewpoint_camera.world_view_transform,
			projmatrix=viewpoint_camera.full_proj_transform,
			sh_degree=self.sh_degree,
			campos=viewpoint_camera.camera_center,
			prefiltered=False,
			debug=False
		)

		rasterizer = GaussianRasterizer(raster_settings=raster_settings)

		means3D = self.get_xyz
		means2D = screenspace_points
		opacity = self.get_opacity
		scales = self.get_scaling
		rotations = self.get_rotation

		# If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
		# from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
		shs = None
		colors_precomp = None
		if override_color is None:
			shs = self.get_features
		else:
			colors_precomp = override_color

		# Rasterize visible Gaussians to image, obtain their radii (on screen).
		rendered_image, radii = rasterizer(
			means3D=means3D,
			means2D=means2D,
			shs=shs,
			colors_precomp=colors_precomp,
			opacities=opacity,
			scales=scales,
			rotations=rotations,
			cov3D_precomp=None)

		# Those Gaussians that were frustum culled or had a radius of 0 were not visible.
		# They will be excluded from value updates used in the splitting criteria.
		return {"rgb": rendered_image,
				"viewspace_points": screenspace_points,
				"visibility_filter": radii > 0,
				"radii": radii}
	

class GsBase(nn.Module):
	def __init__(self, radius: float=2.0, fov_xy: list=[60, 60], white_background: bool=False,
				 size_HW: list=[512, 512], sh_degree: int=0, if_activate:bool=False):
		super().__init__()
		self.radius = radius
		self.fov_xy = fov_xy
		self.white_background = white_background
		self.size_HW = size_HW
		self.gaussian = GaussianModel(sh_degree=sh_degree, if_activate=if_activate)

	def rendering_batch(self, gs_attr, pose=None, return_all=False):
		device = gs_attr['p'].device
		bg_color = [1,1,1] if self.white_background else [0, 0, 0]
		bg_feature = torch.tensor(bg_color, dtype=torch.float32, device=device)
		override_color = None
		rgbs = []
		alls = []
		for idx in range(gs_attr['p'].shape[0]):
			sub_attr = {key: value[idx].contiguous() for key, value in gs_attr.items() if key!='label'}
			shift = torch.mean(sub_attr['p'], dim=0)
			scale = torch.amax(torch.linalg.norm(sub_attr['p']-shift[None], ord=2, dim=-1))
			if pose is None and 'pose' in sub_attr.keys():
				pose = sub_attr['pose']
			view = View(self.radius, self.fov_xy, self.size_HW, scale.cpu(), shift.cpu(), device, random_pose=pose)
			self.gaussian.load_attr_torch(sub_attr)

			render_result = self.gaussian.render(view, bg_feature, override_color=override_color)
			rgbs.append(render_result["rgb"])
			alls.append(render_result)

		if return_all:
			return alls
		else:
			return torch.stack(rgbs, dim=0)  # [B, 3, H, W]
		
	def rendering_one(self, gs_attr, pose=None, return_all=False):
		device = gs_attr['p'].device
		bg_color = [1,1,1] if self.white_background else [0, 0, 0]
		bg_feature = torch.tensor(bg_color, dtype=torch.float32, device=device)
		override_color = None
		shift = torch.mean(gs_attr['p'], dim=0)
		scale = torch.amax(torch.linalg.norm(gs_attr['p']-shift[None], ord=2, dim=-1))
		if pose is None and 'pose' in gs_attr.keys():
			pose = gs_attr['pose']
		view = View(self.radius, self.fov_xy, self.size_HW, scale.cpu(), shift.cpu(), device, random_pose=pose)
		self.gaussian.load_attr_torch(gs_attr)

		render_result = self.gaussian.render(view, bg_feature, override_color=override_color)

		if return_all:
			return render_result
		else:
			return render_result["rgb"]  # [B, 3, H, W]
		
	def rendering(self, gs_attr, pose=None, return_all=False):
		if gs_attr['p'].dim() == 2:
			return self.rendering_one(gs_attr, pose, return_all)
		elif gs_attr['p'].dim() == 3:
			return self.rendering_batch(gs_attr, pose, return_all)
		else:
			print("Wrong Dimension of Input")