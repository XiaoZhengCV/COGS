import os
import json
from typing import Union

from mmengine.registry import DATASETS

@DATASETS.register_module()
class MacgsLoader(object):
	"""Load 3DGS data of MACGS"""
	def __call__(self, root:str, split:str='train', partition: Union[int, float]=0.8):
		print(f'==> Preparing MACGS {split}set...')
		# ply paths
		all_files = []
		if split in ['all', 'train']:
			with open(os.path.join(root, "train.json"), 'r') as f:
				files = [i.split('/') for i in json.load(f)]
				files_dict = {}
				for file in files:
					if file[0] in files_dict:
						files_dict[file[0]].append(os.path.join(root, file[0], file[1]))
					else:
						files_dict[file[0]] = [os.path.join(root, file[0], file[1]),]
				for key, value in files_dict.items():
					all_files += value[:int(len(value)*partition)]
		if split in ['all', 'val', 'test']:
			with open(os.path.join(root, "test.json"), 'r') as f:
				all_files += [os.path.join(root, i) for i in json.load(f)]
		# path: label
		labels_name = self.read_labels(os.path.join(root, "category.txt"))
		labels_dict = {file: int(file.split('/')[-2]) for file in all_files}
		print(f'==> Loaded MACGS {split}set, {len(all_files)} samples')  
		return (all_files, labels_name, labels_dict)  
	
	def read_labels(self, file_path):
		labels_name = {}
		with open(file_path, 'r') as file:
			lines = file.readlines()
		for line in lines:
			index, name = line.strip().split(',', 1)
			labels_name[int(index)] = name
		return labels_name