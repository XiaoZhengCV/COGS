"""Downsample COGS, MACGS, and ModelNet40GS to 1024 Gaussians with FPS."""

import argparse
import multiprocessing as mp
import random
import shutil
from pathlib import Path

import torch
from pointnet2_ops.pointnet2_utils import furthest_point_sample
from tqdm import tqdm

from halo.dataset import load_plydata, save_plydata


DATASETS = {
    "cogs": ("COG", "gs"),
    "macgs": ("MACGS", "raw"),
    "modelnet": ("MODELNET40", "raw"),
}


def sample_chunk(gpu_id, tasks, num_points, force):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    for source, target in tqdm(tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        if target.is_file() and not force:
            continue
        try:
            attributes = load_plydata(str(source), gs_activate=False)
            positions = attributes["p"].contiguous().to(device)
            indices = furthest_point_sample(positions[None], num_points)[0].cpu()
            save_plydata(
                {name: value[indices] for name, value in attributes.items()},
                str(target),
                deactivate=False,
            )
        except Exception as error:
            print(f"\n[GPU {gpu_id}] {source}: {error}")


def copy_metadata(dataset, dataset_dir, source_root, target_root):
    files = {
        "cogs": [dataset_dir / "categories.txt"],
        "macgs": [source_root / name for name in ("category.txt", "train.json", "test.json")],
        "modelnet": [source_root / "label.txt"],
    }[dataset]
    target_root.mkdir(parents=True, exist_ok=True)
    for path in files:
        shutil.copy2(path, target_root / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/xiaoz/code/data"))
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    tasks = []
    for dataset in args.datasets:
        directory, source_name = DATASETS[dataset]
        dataset_dir = data_root / directory
        source_root = dataset_dir / source_name
        target_root = dataset_dir / f"fps_{args.num_points}"
        copy_metadata(dataset, dataset_dir, source_root, target_root)
        tasks.extend(
            (source, target_root / source.relative_to(source_root))
            for source in sorted(source_root.rglob("*.ply"))
        )

    random.Random(42).shuffle(tasks)
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No CUDA GPU is available")

    print(f"Sampling {len(tasks)} files to {args.num_points} points on {gpu_count} GPUs")
    chunks = [tasks[index::gpu_count] for index in range(gpu_count)]
    processes = [
        mp.Process(target=sample_chunk, args=(gpu_id, chunk, args.num_points, args.force))
        for gpu_id, chunk in enumerate(chunks)
        if chunk
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("An FPS worker failed")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
