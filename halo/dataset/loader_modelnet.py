import os
from typing import Union

from mmengine.registry import DATASETS

@DATASETS.register_module()
class ModelNetLoader(object):
	"""Load 3DGS data of ModelNet40"""
	def __call__(self, root:str, split:str='train', partition: Union[int, float]=0.8):
		print(f'==> Preparing ModelNet40 {split}set...')
		# ply paths
		all_files = []
		if split in ['all', 'train']:
			all_files += [os.path.join(root, 'train', i, 'point_cloud.ply') for i in os.listdir(os.path.join(root, 'train'))]
		if split in ['all', 'val', 'test']:
			all_files += [os.path.join(root, 'test', i, 'point_cloud.ply') for i in os.listdir(os.path.join(root, 'test'))]
		# path: label
		labels_name, labels_dict = self.read_labels(os.path.join(root, "label.txt"))
		labels_dict = {file: labels_dict[file.split('/')[-2][:-5]] for file in all_files}
		print(f'==> Loaded ModelNet40 {split}set, {len(all_files)} samples')  
		return (all_files, labels_name, labels_dict)  
	
	def read_labels(self, file_path):
		labels_name = {}
		labels_dict = {}
		with open(file_path, 'r') as file:
			lines = file.readlines()
		for idx, line in enumerate(lines):
			name = line.strip()
			labels_name[idx] = name
			labels_dict[name] = idx
		return labels_name, labels_dict
