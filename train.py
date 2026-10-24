import os
import shutil
import argparse
from pathlib import Path

from mmengine.runner import Runner
from mmengine.config import Config, DictAction
from mmengine.dist import barrier

import halo.dataset 
import halo.metric
import halo.model 
import halo.task
import halo.scheduler

import torch
torch.set_float32_matmul_precision("high")


def shutdown_persistent_workers(runner):
    """Release DataLoader workers so sequential experiments can exit cleanly."""
    for name in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        dataloader = getattr(runner, name, None)
        iterator = getattr(dataloader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            dataloader._iterator = None


def get_test_checkpoint(runner, checkpoint_key):
    """Keep automatic post-training tests within the current experiment."""
    checkpoint_path = runner.message_hub.get_info(checkpoint_key)
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path).expanduser().resolve()
    work_dir = Path(runner.work_dir).resolve()
    if not path.is_relative_to(work_dir):
        raise RuntimeError(
            f'{checkpoint_key} points outside the current work_dir: {path}. '
            f'Expected a checkpoint under {work_dir}. '
            'Use --mode test with an explicit load_from to test external weights.')
    if not path.is_file():
        raise FileNotFoundError(f'Test checkpoint does not exist: {path}')
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model with MMEngine')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '-m', '--mode', 
        choices=['train', 'test', 'val'], 
        default='train', 
        help='Run mode: train, test, or val')
    parser.add_argument(
        '--test-checkpoint',
        choices=['best', 'last'],
        default='best',
        help='Checkpoint tested automatically after training (default: best)')
    parser.add_argument(
        '-o', '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('train_cfg') and 'dynamic_intervals' in cfg.train_cfg:
        intervals = cfg.train_cfg.dynamic_intervals
        if isinstance(intervals, list):
            cfg.train_cfg.dynamic_intervals = [tuple(item) for item in intervals]

    mode = args.mode
    if mode in ['train', 'val', 'test']:
        runner = Runner.from_cfg(cfg)
        try:
            if mode == 'train':
                if runner.rank == 0:
                    src_dir = os.path.join(os.getcwd(), 'halo')
                    dst_dir = os.path.join(runner.log_dir, 'halo')
                    if os.path.exists(src_dir):
                        print(f"📦 正在备份代码：将 {src_dir} 复制到 {dst_dir}")
                        if os.path.exists(dst_dir):
                            shutil.rmtree(dst_dir)
                        shutil.copytree(src_dir, dst_dir)
                    else:
                        print(f"⚠️ 未找到当前目录下的 halo 文件夹，跳过代码备份。")
                runner.train()
                barrier()
                checkpoint_key = f'{args.test_checkpoint}_ckpt'
                checkpoint_path = get_test_checkpoint(runner, checkpoint_key)
                if checkpoint_path is not None:
                    print(
                        f"\n✅ 训练结束！自动加载 {args.test_checkpoint} "
                        f"checkpoint 进行测试: {checkpoint_path}")
                    runner.load_checkpoint(checkpoint_path)
                else:
                    print(
                        f"\n⚠️ 未找到 {args.test_checkpoint} checkpoint 记录，"
                        "将直接使用训练结束时的当前权重进行测试。")
                runner.test()
            else:
                getattr(runner, mode)()
        finally:
            shutdown_persistent_workers(runner)


if __name__ == '__main__':
    main()
