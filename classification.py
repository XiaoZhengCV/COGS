"""Run COGS classification experiments, assigning one job to each GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import sys
from threading import Lock


ROOT = Path(__file__).resolve().parent
MODELS = (
    'pointnet', 'dgcnn', 'ptv1', 'gaussianmae', 'dela',
    'halo_base', 'halo_large', 'halo_test',
)
INPUT_TYPES = ('p', 'po', 'pc', 'poc', 'psq', 'posq', 'psqc', 'posqc')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--inputs', nargs='+', choices=INPUT_TYPES, default=['posqc'])
    parser.add_argument('--gpus', nargs='+', default=[str(i) for i in range(8)])
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--start-run', type=int, default=1,
                        help='Output directory index; the training seed remains 42.')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs' / 'cogs')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without training.')
    args = parser.parse_args()
    if args.runs < 1 or args.start_run < 1:
        parser.error('--runs and --start-run must be >= 1')
    for name in ('models', 'inputs', 'gpus'):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f'--{name} must not contain duplicates')
    args.output_root = args.output_root.expanduser().resolve()
    if args.data_root is not None:
        args.data_root = args.data_root.expanduser().resolve()
    return args


def build_commands(args):
    commands = []
    for model in args.models:
        for input_type in args.inputs:
            for run in range(args.start_run, args.start_run + args.runs):
                command = [
                    sys.executable, str(ROOT / 'train.py'),
                    str(ROOT / 'config' / 'cogs' / f'{model}.yml'), '-o',
                    f'work_dir={args.output_root / f"{input_type}_{run}" / model}',
                    f'model.backbone.input_type={input_type}',
                ]
                for split in ('train', 'val', 'test'):
                    command.append(f'{split}_dataloader.dataset.input_type={input_type}')
                    if args.data_root is not None:
                        command.append(f'{split}_dataloader.dataset.root={args.data_root.resolve()}')
                commands.append(command)
    return commands


def run_commands(commands, gpus):
    tasks = Queue()
    for command in commands:
        tasks.put(command)
    failures = []
    lock = Lock()

    def worker(gpu_id):
        while True:
            try:
                command = tasks.get_nowait()
            except Empty:
                return
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f'[GPU {gpu_id}] {shlex.join(command)}', flush=True)
            try:
                result = subprocess.run(command, cwd=ROOT, env=env, check=False)
                if result.returncode:
                    with lock:
                        failures.append((command, result.returncode))
                    print(f'[GPU {gpu_id}] FAILED: exit code {result.returncode}', flush=True)
                else:
                    print(f'[GPU {gpu_id}] Completed', flush=True)
            except OSError as error:
                with lock:
                    failures.append((command, str(error)))
                print(f'[GPU {gpu_id}] FAILED: {error}', flush=True)
            finally:
                tasks.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    return failures


def main():
    args = parse_args()
    commands = build_commands(args)
    if args.dry_run:
        for index, command in enumerate(commands):
            print(f'CUDA_VISIBLE_DEVICES={args.gpus[index % len(args.gpus)]} {shlex.join(command)}')
        return 0
    failures = run_commands(commands, args.gpus)
    print(f'Completed {len(commands) - len(failures)}/{len(commands)} experiments.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
