from typing import List, Union

import torch
import torch.nn as nn

from .cutils import knn_edge_maxpooling
from .utils import index_points, calc_pwd, ConvBNAct, ConvBN, LinearBNAct

from mmengine.registry import MODELS


@MODELS.register_module()
class HALO(nn.Module):
	"""Model for 3DGS classification"""
	def __init__(self, num_classes: int, input_type: str='pocsq', 
			  	 out_channels: List=[128, 256, 512], group_channels: List=[32, 32, 32], 
				 k_neighbors: List=[20, 20, 20], n_points: List=[1024, 256, 64], 
				 ratio: Union[int, float]=0.5, emb_dims: int=2048):
		super().__init__()
		self.input_type = input_type 

		self.stages = len(out_channels)
		out_channels.append(emb_dims)
		self.fe_list = nn.ModuleList()
		self.ha_list = nn.ModuleList()
		for i in range(self.stages):
			fe = FE(out_channels[i], group_channels[i], n_points[i], k_neighbors[i], input_type)
			self.fe_list.append(fe)
			ha = HA(out_channels[i], out_channels[i+1], ratio)
			self.ha_list.append(ha)

		self.classifier = nn.Sequential(
			LinearBNAct(emb_dims, 512, nn.ReLU(inplace=True)),
			nn.Dropout(0.5),
			LinearBNAct(512, 256, nn.ReLU(inplace=True)),
			nn.Dropout(0.5),
			nn.Linear(256, num_classes)
		)

	def forward(self, gs_attr):
		gs_data = gs_attr['gs_data'].transpose(-1, -2)
		xyz = gs_data[:, :3, :] # [b,3,n]
		
		knn_xyz = calc_pwd(xyz)	# [b,n,n]
		x = None

		for i in range(self.stages):
			x, idx= self.fe_list[i](knn_xyz, gs_data, x)
			x = self.ha_list[i](x, idx)  # [b,d,g]

		x = x.max(dim=-1)[0]
		x = self.classifier(x)
		return x


class FE(nn.Module):
	def __init__(self, out_channel, group_channel, n_point, kneighbors, input_type):
		super().__init__()
		self.n_point = n_point
		self.kneighbors = kneighbors
		self.input_type = input_type

		self.group_xyz_embed = LinearEmbed(kneighbors, group_channel)
		self.xyz_embed = ConvEmbed(3, group_channel, dim=3)
		self.xyz_proj = ConvBN(group_channel*4, out_channel, dim=3)

		self.group_other_embed = nn.ModuleDict()
		self.slice = {}
		other_dim = 0
		if 'o' in self.input_type:
			self.group_other_embed['o'] = LinearEmbed(kneighbors, group_channel//2)
			self.slice['o'] = slice(other_dim, other_dim+1)
			other_dim += 1
		if 'c' in self.input_type:
			self.group_other_embed['c'] = LinearEmbed(kneighbors, group_channel//2)
			self.slice['c'] = slice(other_dim, other_dim+3)
			other_dim += 3
		if 's' in self.input_type:
			self.group_other_embed['s'] = LinearEmbed(kneighbors, group_channel//2)
			self.slice['s'] = slice(other_dim, other_dim+3)
			other_dim += 3
		if 'q' in self.input_type:
			self.group_other_embed['q'] = LinearEmbed(kneighbors, group_channel//2)
			self.slice['q'] = slice(other_dim, other_dim+3)
			other_dim += 3
		if 'n' in self.input_type:
			self.group_other_embed['n'] = LinearEmbed(kneighbors, group_channel//2)
			self.slice['n'] = slice(other_dim, other_dim+3)
			other_dim += 3
		if other_dim>0:
			self.other_embed = ConvEmbed(other_dim, group_channel//2, dim=3)
			self.other_proj = ConvBN(group_channel//2*(other_dim+1), out_channel, dim=3)

		self.other_dim = other_dim

	def forward(self, knn_xyz, point_feat, x):
		S = self.n_point
		K = self.kneighbors
		_, idx = knn_xyz[:, :S, :S].topk(k=K, dim=-1, largest=False, sorted=True)

		# Gather xyz and all attributes together, then split them for each branch.
		grouped_neighbors = index_points(point_feat, idx)
		grouped_neighbors = grouped_neighbors - point_feat[:, :, :S].unsqueeze(-1)
		xyz = point_feat[:, :3, :]
		other = point_feat[:, 3:, :]

		xyz_grouped_feat = self.group_xyz_embed(grouped_neighbors[:, :3, :, :])
		xyz_feat = self.xyz_embed(xyz[:, :, :S])
		xyz_feat = torch.cat([xyz_feat, xyz_grouped_feat], dim=1)
		grouped_feat = self.xyz_proj(xyz_feat)

		if self.other_dim>0:
			other_grouped_feat = []
			for i in self.input_type:
				if i in 'ocsqn':
					attr_slice = self.slice[i]
					feat = self.group_other_embed[i](
						grouped_neighbors[:, 3 + attr_slice.start:3 + attr_slice.stop, :, :]
					)
					other_grouped_feat.append(feat)
			other_grouped_feat = torch.cat(other_grouped_feat, dim=1)

			other_feat = self.other_embed(other[:, :, :S])
			other_feat = torch.cat([other_grouped_feat, other_feat], dim=1)
			other_feat = self.other_proj(other_feat)
			
			grouped_feat = grouped_feat + other_feat

		if x==None:
			x = grouped_feat 
		else:
			x = grouped_feat + x[:, :, :S]
		
		return x, idx
	
class HA(nn.Module):
	def __init__(self, in_channel, out_channel, ratio=0.5):
		super().__init__()

		mid_channel = int(in_channel * ratio)
		self.feat_transform = nn.Sequential(
			ConvBNAct(in_channel, mid_channel, nn.ReLU(inplace=True), dim=3),
			ConvBNAct(mid_channel, mid_channel, nn.ReLU(inplace=True), dim=3),
			ConvBNAct(mid_channel, in_channel, nn.ReLU(inplace=True), dim=3),
		)
		self.feat_propagate_1 = FDT(in_channel, in_channel)
		self.skip = ConvBN(in_channel, out_channel, dim=3)
		self.feat_propagate_2 = FDT(in_channel, out_channel)

	def forward(self, x, idx):
		x = x + self.feat_transform(x)
		x = x + self.feat_propagate_1(x, idx)
		x = self.skip(x) + self.feat_propagate_2(x, idx)
		return x
	

class FDT(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()
		self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
		self.bn = nn.BatchNorm1d(out_dim)
	
	def forward(self, x, idx):
		x = self.conv(x)
		x = knn_edge_maxpooling(x.transpose(2, 1).contiguous(), idx, self.training).transpose(2, 1)
		x = self.bn(x)
		return x
	

class ConvEmbed(nn.Module):
	def __init__(self, in_channel, out_channel, dim=3):
		super().__init__()
		self.net = nn.Sequential(
			ConvBNAct(in_channel, out_channel//2, nn.ReLU(inplace=True), dim),
			ConvBNAct(out_channel//2, out_channel, nn.ReLU(inplace=True), dim),
		)

	def forward(self, x):
		return self.net(x)
	

class LinearEmbed(nn.Module):
	def __init__(self, in_channel, out_channel):
		super().__init__()
		self.net = nn.Sequential(
			LinearBNAct(in_channel, out_channel//2, nn.ReLU(inplace=True)),
			LinearBNAct(out_channel//2, out_channel, nn.ReLU(inplace=True)),
		)

	def forward(self, x):
		B, C, N, K = x.shape
		x = x.reshape(-1, K)
		return self.net(x).reshape(B, C, N, -1).permute(0, 1, 3, 2).reshape(B, -1, N)
