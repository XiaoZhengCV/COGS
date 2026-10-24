import os
import json
from typing import Union

from mmengine.registry import DATASETS

SPLIT_FILES = {
    100: ("train_benchmark.json", "test_benchmark.json"),
    0.8: ("train_pretrain.json", "test_pretrain.json"),
}


@DATASETS.register_module()
class COGSLoader(object):
    """Read fixed COGS splits: 100 training objects/class or 80%/class."""

    def __call__(self, root: str, split: str = 'train', partition: Union[int, float] = 0.8):
        if partition not in SPLIT_FILES:
            raise ValueError(
                "COGS partition must be 100 (benchmark) or 0.8 (pretraining)."
            )

        train_name, test_name = SPLIT_FILES[partition]
        train_json = os.path.join(root, train_name)
        test_json = os.path.join(root, test_name)
        for path in (train_json, test_json):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"COGS split file not found: {path}")

        print(f'==> Preparing COGS {split}set...')

        # Split files contain paths relative to the dataset root.
        all_files = []
        if split in ['all', 'train']:
            with open(train_json, 'r', encoding='utf-8') as f:
                all_files += [os.path.join(root, i) for i in json.load(f)]
        if split in ['all', 'val', 'test']:
            with open(test_json, 'r', encoding='utf-8') as f:
                all_files += [os.path.join(root, i) for i in json.load(f)]

        labels_name, labels_dict = self.read_labels(os.path.join(root, "categories.txt"))
        labels_dict = {file: int(file.split('/')[-2][:2]) for file in all_files}

        print(f'==> Loaded COGS {split}set, {len(all_files)} samples')
        return (all_files, labels_name, labels_dict)

    def read_labels(self, file_path):
        labels_name = {}
        labels_dict = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            idx, name = line.strip().split(':')
            labels_name[int(idx)] = name.strip().replace('_', ' ')
            labels_dict[name] = idx
        return labels_name, labels_dict
