## COGS
The official implementation of our work: **COGS: A Diagnostic Benchmark for Attribute-Aware Object-Level 3D Gaussian Splatting Perception**

## Environment Setup

>Python: 3.11  
>Pytorch: 2.5.1  
>CUDA: 11.8  

The checked environment uses Python 3.11, PyTorch 2.5.1 with CUDA 11.8 and MMEngine 0.10.7. 
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.10.7 einops e3nn plyfile scikit-learn ninja pandas ipykernel h5py future tensorboard timm iopath
```
Other dependencies:
- Pytorch3d==0.7.8: [Link](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- pointops: [Link](https://github.com/Pointcept/Pointcept/tree/main/libs/pointops)
- pointnet2_ops: [Link](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2_ops_lib)

> If you encounter No module named 'torch' error, use pip install . --no-build-isolation  
> It will take some time to train the model for the first time, since [cutils](./halo/model/cutils) need to be build.

## Data and weights
[COGS](https://huggingface.co/datasets/XiaoZhengCV/COGS)
Set the `train_dataloader.dataset.root` in the [default.yml](./config/cogs/default.yml) to your downloaded data directory.  
Gaussian-MAE additionally needs pretrained checkpoint following [ShapeSplat](https://github.com/qimaqi/ShapeSplat-Gaussian_MAE)

## Run experiments

Run commands from this directory with the intended Python environment activated.
For a single HALO-B run:

```bash
python train.py config/cogs/halo_base.yml \
  -o work_dir=outputs/cogs/posqc_1/halo_base
```

Available configs are `pointnet`, `dgcnn`, `ptv1`, `gaussianmae`, `dela`,
`halo_base`, `halo_large` and `halo_test` (shared attribute processing ablation).
Backbone and dataset input attributes must agree. Supported benchmark inputs are
`p`, `po`, `pc`, `poc`, `psq`, `posq`, `psqc`, `posqc`.

The launcher assigns one experiment to each listed GPU. Preview commands first:

```bash
python classification.py --models halo_base halo_test --inputs posqc --gpus 0 1 --dry-run
```

Remove `--dry-run` to execute them. The default launcher runs all eight configs
with POSQC once on GPU IDs 0 through 7. For the seven-model attribute diagnosis:

```bash
python classification.py \
  --models pointnet dgcnn ptv1 gaussianmae dela halo_base halo_large \
  --inputs p po pc poc psq posq psqc posqc --gpus 0 1 2 3 4 5 6 7 --dry-run
```

Use `--data-root /path/to/fps_1024` to override the root for all three dataloaders,
and `--output-root /path/to/results` to choose the result directory. `--runs` and
`--start-run` control output directory indices, not random seeds; configs retain
training seed 42. Failed jobs are reported and the launcher exits nonzero if any
job fails. Importing the launcher does not start training.

Models train for 200 epochs with 1,024 primitives and batch size 32 per GPU.
Each classification job uses one GPU. Gaussian-MAE retains its separate optimizer
and scheduler settings; all existing architecture, preprocessing and optimization
settings are preserved in their configs.

## Evaluate a checkpoint

```bash
python train.py config/cogs/halo_base.yml --mode test \
  -o work_dir=outputs/cogs/evaluation/halo_base load_from=/path/to/checkpoint.pth
```

After training, `train.py` automatically evaluates the best checkpoint by default
(`--test-checkpoint last` selects the last checkpoint). Automatic evaluation rejects
checkpoint paths outside the current experiment directory, preventing stale paths
from silently loading another experiment's weights. Explicit `--mode test` can
evaluate external checkpoints.
