#!/bin/bash
#SBATCH -J omol25_symphony
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_symphony-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/train_symphony-%j.err
# Symphony baseline (autoregressive, JAX/GPU), bond-free, OMol25 50k.
# Real working set is small; jax mem capped so it fits the 2080ti (11 GB).
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PY=$SCR/envs/symphony/bin/python
NV=$SCR/envs/symphony/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cuda_cupti/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9   # fit jax in the 2080ti
export PYTHONUNBUFFERED=1
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO/baselines/symphony"
$PY -u -m symphony --config configs/omol25/full.py \
    --workdir "$SCR/runs/omol25_50k_symphony" --use_wandb=False
date
