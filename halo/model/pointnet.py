import torch
import torch.nn as nn

from .utils import ConvBN, ConvBNAct, LinearBNAct

from mmengine.registry import MODELS


# Adaption of the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


@MODELS.register_module()
class PointNet(nn.Module):
	"""Basic PointNet Model for point cloud classification, apply to 3DGS"""
	def __init__(self, num_classes: int, input_type: str='pocsq'):
		super().__init__()
		self.input_type = input_type
		self.mat_diff_loss_scale = 0.001

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

		self.stn = STNkd(input_dim, k=3)
		self.conv1 = ConvBNAct(input_dim, 64, nn.ReLU(), dim=3, bias=True)
		self.fstn = STNkd(channel=64, k=64)
		self.conv2 = ConvBNAct(64, 128, nn.ReLU(), dim=3, bias=True)
		self.conv3 = ConvBN(128, 1024, dim=3, bias=True)

		self.classifier = nn.Sequential(
			LinearBNAct(1024, 512, nn.ReLU(), bias=True),
			nn.Linear(512, 256),
			nn.Dropout(0.4),
			nn.BatchNorm1d(256), 
			nn.ReLU(),
			nn.Linear(256, num_classes)
		)

	def forward(self, gs_attr):
		x = gs_attr['gs_data'].transpose(-1, -2)

		x, trans = self.stn(x)
		x = self.conv1(x)
		x, trans_feat = self.fstn(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = torch.max(x, dim=2, keepdim=False)[0]
		x = x.view(-1, 1024)
		x = self.classifier(x)

		if self.training:
			mat_diff_loss = feature_transform_reguliarzer(trans_feat)
			mat_diff_loss = mat_diff_loss * self.mat_diff_loss_scale
			return x, mat_diff_loss
		else:
			return x


class STNkd(nn.Module):
	"""Spatial Transformer Network for 3D input alignment"""
	def __init__(self, channel:int=3, k: int=3):
		super().__init__()
		self.k = k

		self.conv = nn.Sequential(
			ConvBNAct(channel, 64, nn.ReLU(), dim=3, bias=True),
			ConvBNAct(64, 128, nn.ReLU(), dim=3, bias=True),
			ConvBNAct(128, 1024, nn.ReLU(), dim=3, bias=True),
		)
		self.fc = nn.Sequential(
			LinearBNAct(1024, 512, nn.ReLU(), bias=True),
			LinearBNAct(512, 256, nn.ReLU(), bias=True),
			nn.Linear(256, k*k)
		)

	def forward(self, x):
		trans = self.conv(x)        
		trans = torch.max(trans, dim=-1, keepdim=False)[0]      
		trans = self.fc(trans)                               
		trans = trans.view(-1, self.k, self.k)

		# Add identity transformation
		iden = torch.eye(self.k, dtype=torch.float32, device=trans.device)
		trans = trans + iden                   

		x = x.transpose(2, 1)
		feature = x[:, :, self.k:]
		x = x[:, :, :self.k]
		x = torch.bmm(x, trans)
		x = torch.cat([x, feature], dim=2)
		x = x.transpose(2, 1)

		return x, trans
