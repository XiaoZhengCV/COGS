from collections import OrderedDict

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from pointnet2_ops.pointnet2_utils import furthest_point_sample

from .utils import knn

from mmengine.registry import MODELS


# Adaption of the implementation of https://github.com/qimaqi/ShapeSplat-Gaussian_MAE/blob/main/models/Gaussian_MAE.py


CANONICAL_GS_INPUT_TYPE = "posqc"
GS_ATTR_DIMS = {
	"p": 3,
	"o": 1,
	"s": 3,
	"q": 4,
	"c": 3,
	"n": 3,
}
CANONICAL_GS_ATTR_SLICES = {
	"p": slice(0, 3),
	"o": slice(3, 4),
	"s": slice(4, 7),
	"q": slice(7, 11),
	"c": slice(11, 14),
}
CANONICAL_GS_INPUT_DIM = sum(
	GS_ATTR_DIMS[attr] for attr in CANONICAL_GS_INPUT_TYPE
)


def index_points(points, idx):
	"""
	points: [B, N, C]
	idx: [B, S] or [B, N, K]
	returns: [B, S, C] or [B, N, K, C]
	"""
	_, _, C = points.shape
	if idx.dim() == 2:  # [B, S]
		idx_expanded = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, S, C]
	elif idx.dim() == 3:  # [B, N, K]
		# reshape points to [B, N, 1, C] so it can be gathered using dim=2
		points = points.unsqueeze(2).expand(-1, -1, idx.shape[2], -1)  # [B, N, K, C]
		idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)         # [B, N, K, C]
	return torch.gather(points, dim=1, index=idx_expanded) 


def _validate_input_type(input_type):
	if not isinstance(input_type, str) or len(input_type) == 0:
		raise ValueError("input_type must be a non-empty string, e.g. 'posqc'.")
	if len(set(input_type)) != len(input_type):
		raise ValueError(f"input_type contains duplicated attributes: {input_type}")
	unknown = set(input_type) - set(GS_ATTR_DIMS)
	if unknown:
		raise ValueError(
			f"Unknown input_type attributes {sorted(unknown)}; "
			f"allowed attributes are {sorted(GS_ATTR_DIMS)}"
		)
	if "p" not in input_type:
		raise ValueError("GaussianMAE requires 'p' in input_type for FPS/KNN grouping.")


def _compact_input_dim(input_type):
	return sum(GS_ATTR_DIMS[attr] for attr in input_type)


def canonicalize_gs_data(gs_data, input_type):
	"""Convert compact input_type features to fixed p,o,s,q,c layout."""
	_validate_input_type(input_type)
	expected_dim = _compact_input_dim(input_type)
	if gs_data.shape[-1] != expected_dim:
		raise ValueError(
			f"gs_data last dim mismatch for input_type='{input_type}': "
			f"expected {expected_dim}, got {gs_data.shape[-1]}."
		)
	if input_type == CANONICAL_GS_INPUT_TYPE:
		return gs_data

	canonical = gs_data.new_zeros((*gs_data.shape[:-1], CANONICAL_GS_INPUT_DIM))
	in_offset = 0
	for attr in input_type:
		attr_dim = GS_ATTR_DIMS[attr]
		if attr in CANONICAL_GS_ATTR_SLICES:
			canonical[..., CANONICAL_GS_ATTR_SLICES[attr]] = gs_data[
				..., in_offset : in_offset + attr_dim
			]
		in_offset += attr_dim
	return canonical


class GaussianMAEBackbone(nn.Module):
	"""The feature extraction backbone of GaussianMAE."""
	def __init__(self, input_type: str='posqc', trans_dim: int=384, depth: int=12,
				 drop_path_rate: float=0.1, num_heads: int=6, group_size: int=32, num_group: int=128):
		super().__init__()

		self.trans_dim = trans_dim
		self.attribute = input_type
		_validate_input_type(self.attribute)
		self.pos_feature_dim = [0, 1, 2] # Position features used for embedding

		self.group_divider = Group(
			num_group=num_group,
			group_size=group_size
		)

		self.encoder = Encoder(
			encoder_channel=trans_dim,
			attribute=self.attribute
		)

		self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
		self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

		self.pos_embed = nn.Sequential(
			nn.Linear(len(self.pos_feature_dim), 128),
			nn.GELU(),
			nn.Linear(128, self.trans_dim),
		)

		dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
		self.blocks = TransformerEncoder(
			embed_dim=self.trans_dim,
			depth=depth,
			drop_path_rate=dpr,
			num_heads=num_heads,
		)

		self.norm = nn.LayerNorm(self.trans_dim)

	def forward(self, gs_attr):
		pts = canonicalize_gs_data(gs_attr['gs_data'], self.attribute)

		neighborhood, center = self.group_divider(pts)  # [B, G, M, K], [B, G, K]
		group_input_tokens = self.encoder(neighborhood)  # B G 384

		cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
		cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

		center_pos = center[..., self.pos_feature_dim]
		pos = self.pos_embed(center_pos)

		x = torch.cat((cls_tokens, group_input_tokens), dim=1)
		pos = torch.cat((cls_pos, pos), dim=1)

		x = self.blocks(x, pos)
		x = self.norm(x)
		concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
		
		return concat_f




@MODELS.register_module()
class GaussianMAE(nn.Module):
	"""Gaussian MAE for 3DGS"""
	def __init__(self, num_classes: int, input_type: str='pocsq', head_type: str='full',
			  	 trans_dim: int=384, depth: int=12, drop_path_rate: float=0.1, num_heads: int=6,
				 group_size: int=32, num_group: int=128, ckpt_path: str=''):
		super().__init__()

		self.module = GaussianMAEBackbone(
			input_type=input_type,
			trans_dim=trans_dim,
			depth=depth,
			drop_path_rate=drop_path_rate,
			num_heads=num_heads,
			group_size=group_size,
			num_group=num_group
		)

		if ckpt_path != '':
			self._load_pretrained_ckpt(ckpt_path)
		else:
			self.apply(self._init_weights)

		if head_type == "linear":
			self.cls_head_finetune = nn.Sequential(
				nn.Linear(trans_dim * 2, num_classes)
			)
		else:
			self.cls_head_finetune = nn.Sequential(
				nn.Linear(trans_dim * 2, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Linear(256, 256),
				nn.BatchNorm1d(256),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Linear(256, num_classes),
			)

	def _init_weights(self, m):
		trunc_normal_(self.module.cls_token, std=0.02)
		trunc_normal_(self.module.cls_pos, std=0.02)
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=0.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Conv1d):
			trunc_normal_(m.weight, std=0.02)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def _load_pretrained_ckpt(self, ckpt_path):
		checkpoint = torch.load(ckpt_path, map_location='cpu')
		pretrained_state_dict = checkpoint['base_model']
		
		model_state_dict = self.state_dict()
		filtered_pretrained_dict = OrderedDict()

		for k, v in pretrained_state_dict.items():
			if 'cls_head_finetune' in k:
				k = k.replace("module.", "")
			if 'MAE_encoder' in k:
				k = k.replace("MAE_encoder.", "")
			if k in model_state_dict and v.shape == model_state_dict[k].shape:
				filtered_pretrained_dict[k] = v
		
		self.load_state_dict(filtered_pretrained_dict, strict=False)

	def forward(self, gs_attr):
		concat_f = self.module(gs_attr)
		ret = self.cls_head_finetune(concat_f)
		return ret


class TransformerEncoder(nn.Module):
	def __init__(
		self,
		embed_dim=768,
		depth=4,
		num_heads=12,
		mlp_ratio=4.0,
		qkv_bias=False,
		qk_scale=None,
		drop_rate=0.0,
		attn_drop_rate=0.0,
		drop_path_rate=0.0,
	):
		super().__init__()

		self.blocks = nn.ModuleList(
			[
				Block(
					dim=embed_dim,
					num_heads=num_heads,
					mlp_ratio=mlp_ratio,
					qkv_bias=qkv_bias,
					qk_scale=qk_scale,
					drop=drop_rate,
					attn_drop=attn_drop_rate,
					drop_path=(
						drop_path_rate[i]
						if isinstance(drop_path_rate, list)
						else drop_path_rate
					),
				)
				for i in range(depth)
			]
		)

	def forward(self, x, pos):
		for _, block in enumerate(self.blocks):
			x = block(x + pos)
		return x


class Block(nn.Module):
	def __init__(
		self,
		dim,
		num_heads,
		mlp_ratio=4.0,
		qkv_bias=False,
		qk_scale=None,
		drop=0.0,
		attn_drop=0.0,
		drop_path=0.0,
		act_layer=nn.GELU,
		norm_layer=nn.LayerNorm,
	):
		super().__init__()
		self.norm1 = norm_layer(dim)

		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = Mlp(
			in_features=dim,
			hidden_features=mlp_hidden_dim,
			act_layer=act_layer,
			drop=drop,
		)

		self.attn = Attention(
			dim,
			num_heads=num_heads,
			qkv_bias=qkv_bias,
			qk_scale=qk_scale,
			attn_drop=attn_drop,
			proj_drop=drop,
		)

	def forward(self, x):
		x = x + self.drop_path(self.attn(self.norm1(x)))
		x = x + self.drop_path(self.mlp(self.norm2(x)))
		return x


class Mlp(nn.Module):
	def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features
		self.fc1 = nn.Linear(in_features, hidden_features)
		self.act = act_layer()
		self.fc2 = nn.Linear(hidden_features, out_features)
		self.drop = nn.Dropout(drop)

	def forward(self, x):
		x = self.fc1(x)
		x = self.act(x)
		x = self.drop(x)
		x = self.fc2(x)
		x = self.drop(x)
		return x


class Attention(nn.Module):
	def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
		super().__init__()
		self.num_heads = num_heads
		head_dim = dim // num_heads
		# NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
		self.scale = qk_scale or head_dim**-0.5
		self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
		self.attn_drop = nn.Dropout(attn_drop)
		self.proj = nn.Linear(dim, dim)
		self.proj_drop = nn.Dropout(proj_drop)

	def forward(self, x):
		B, N, C = x.shape
		qkv = (
			self.qkv(x)
			.reshape(B, N, 3, self.num_heads, C // self.num_heads)
			.permute(2, 0, 3, 1, 4)
		)
		# make torchscript happy (cannot use tensor as tuple)
		q, k, v = qkv[0], qkv[1], qkv[2]

		attn = (q @ k.transpose(-2, -1)) * self.scale
		attn = attn.softmax(dim=-1)
		attn = self.attn_drop(attn)

		x = (attn @ v).transpose(1, 2).reshape(B, N, C)
		x = self.proj(x)
		x = self.proj_drop(x)
		return x


class Encoder(nn.Module):  
	def __init__(self, encoder_channel, attribute="pocsq"):
		super().__init__()
		self.encoder_channel = encoder_channel
		self.attribute = CANONICAL_GS_INPUT_TYPE
		input_dim = CANONICAL_GS_INPUT_DIM

		self.first_conv = nn.Sequential(
			nn.Conv1d(input_dim, 128, 1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Conv1d(128, 256, 1),
		)
		self.second_conv = nn.Sequential(
			nn.Conv1d(512, 512, 1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, self.encoder_channel, 1),
		)

	def forward(self, point_groups):
		"""
		point_groups : B G M 3 | gs: B G M K
		B: Batch size
		G: Number of groups
		O: Number of potential neighbors per point
		M: Number of points per group
		K: Number of dimensions per point (depends on attributes)
		-----------------
		feature_global : B G C
		"""
		bs, g, n, _ = point_groups.shape
		point_groups = point_groups.reshape(bs * g, n, -1).contiguous()  # (BG, M, K)

		# encoder
		feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 m
		feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
		feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 m
		feature = self.second_conv(feature)  # BG 384 n
		feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 384
		return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
	def __init__(self, num_group, group_size):
		super().__init__()
		self.num_group = num_group
		self.group_size = group_size 
		self.attribute_index = [0, 1, 2]

	def forward(self, gs_attr):
		"""
		input: B N 3 or B N K
		---------------------------
		output: B G M 3 or (B G O 3) if select potential neighbors
		center : B G 3
		"""
		center = fps_gs(gs_attr, self.num_group)  # B G K
		center_group = center[..., self.attribute_index]
		xyz_group = gs_attr[..., self.attribute_index]
		idx, _ = knn(center_group, xyz_group, k=self.group_size, transpose=True)  # B G M

		neighborhood = index_points(gs_attr, idx)
		neighborhood[..., :3] = neighborhood[..., :3] - center.unsqueeze(2)[..., :3]

		return neighborhood, center
	

def fps_gs(gs_attr, number, return_idx=False):
	fps_idx = furthest_point_sample(gs_attr[:, :, :3].clone(), number).long()
	if return_idx:
		return fps_idx
	fps_data = index_points(gs_attr, fps_idx)  # [B, g, d]
	return fps_data
