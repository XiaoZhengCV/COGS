import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_graph_feature, ConvBNAct, LinearBNAct

from mmengine.registry import MODELS


# Adaption of the implementation of https://github.com/antao97/dgcnn.pytorch/blob/master/model.py


@MODELS.register_module()
class DGCNN(nn.Module):
	"""Basic DGCNN Model for point cloud classification, apply to 3DGS"""
	def __init__(self, n_knn: int, emb_dims:int, dropout: float, num_classes: int, input_type: str='pocsq'):
		super().__init__()
		self.k = n_knn
		self.input_type = input_type

		input_dim = 3
		if 'o' in self.input_type:
			input_dim += 1
		if 'c' in self.input_type:
			input_dim += 3
		if 's' in self.input_type:
			input_dim += 3
		if 'q' in self.input_type:
			input_dim += 4
		if 'n' in self.input_type:
			input_dim += 3

		self.conv1 = ConvBNAct(input_dim*2, 64, nn.LeakyReLU(negative_slope=0.2), dim=4)
		self.conv2 = ConvBNAct(64*2, 64, nn.LeakyReLU(negative_slope=0.2), dim=4)
		self.conv3 = ConvBNAct(64*2, 128, nn.LeakyReLU(negative_slope=0.2), dim=4)
		self.conv4 = ConvBNAct(128*2, 256, nn.LeakyReLU(negative_slope=0.2), dim=4)
		self.conv5 = ConvBNAct(512, emb_dims, nn.LeakyReLU(negative_slope=0.2), dim=3)

		self.classifier = nn.Sequential(
			LinearBNAct(emb_dims*2, 512, nn.LeakyReLU(negative_slope=0.2)),
			nn.Dropout(p=dropout),
			LinearBNAct(512, 256, nn.LeakyReLU(negative_slope=0.2), bias=True),
			nn.Dropout(p=dropout),
			nn.Linear(256, num_classes)
		)
		

	def forward(self, gs_attr):
		x = self.get_feature(gs_attr)
		batch_size = x.shape[0]
		x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
		x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
		x = torch.cat([x1, x2], dim=1)              # (batch_size, emb_dims*2)
		x = self.classifier(x)                      # (batch_size, emb_dims*2) -> (batch_size, 512) -> (batch_size, 256) -> (batch_size, output_channels)
		
		return x
	
	def get_feature(self, gs_attr):
		x = gs_attr['gs_data'].transpose(-1, -2).contiguous()

		x = get_graph_feature(x, k=self.k)    # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
		x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
		x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

		x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
		x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
		x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

		x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
		x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
		x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

		x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
		x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
		x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

		x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256=512, num_points)
		x = self.conv5(x)            			# (batch_size, 512, num_points) -> (batch_size, emb_dims, num_points)
		return x
