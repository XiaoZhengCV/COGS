from typing import Union

import torch
from torch.utils.data import Dataset

from .utils_load import load_plydata, transform

from mmengine.registry import DATASETS


@DATASETS.register_module()
class GaussianDataset(Dataset):
	"""load gaussian splatting data for classification"""
	def __init__(self, loader: dict, root: str, transform: list, num_points: int=1024, 
				 input_type: str='pocsq', gs_activate: bool=True, sh_degree: int=0,
				 split: str='train', partition: Union[int, float]=0.8):
		self.all_data = {}
		self.data_dir = root
		self.split = split  
		self.transform = transform
		self.num_points = num_points
		self.gs_activate = gs_activate
		self.input_type = input_type
		self.sh_degree = sh_degree
		
		loader = DATASETS.build(loader)
		self.metainfo = dict(name='GaussianDataset', loader=str(loader), data_dir=root, split=split, num_points=num_points)
		self.files, self.labels_name, self.labels_dict = loader(root, split, partition)

	def __getitem__(self, index):
		gs_attr = load_plydata(self.files[index], self.gs_activate, self.sh_degree, self.input_type)
		if self.split=='train' or 'train' in self.split:
			gs_attr = transform(gs_attr, self.transform)
		else:
			with torch.random.fork_rng(devices=[]): 
				torch.manual_seed(index + 42)
				gs_attr = transform(gs_attr, self.transform)
		data = {}
		data['gs_data'] = torch.cat([gs_attr[i] for i in self.input_type], dim=-1)
		data['label'] = self.labels_dict[self.files[index]]
		return data
	
	def get_name(self, index):
		label = self.labels_dict[self.files[index]]
		return self.labels_name[label]

	def __len__(self):
		return len(self.files)