#!/bin/bash
#SBATCH -J omol25_symphony
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 2-00:00:00 --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/symphony-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/symphony-%j.err
# Symphony baseline (autoregressive, JAX), bond-free, OMol25. 存储全在 woo_lab。
set -eo pipefail
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
PY=$STORE/envs/symphony/bin/python
NV=$STORE/envs/symphony/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/cublas/lib:$NV/cuda_cupti/lib:${LD_LIBRARY_PATH:-}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH=$STORE/baselines/symphony
export PYTHONUNBUFFERED=1
CONFIG="${1:-configs/omol25/full.py}"
WORKDIR="${2:-$STORE/runs/omol25_symphony}"   # checkpoint 全部落在 woo_lab 大盘
echo "host $(hostname); $(nvidia-smi -L)"; echo "config=$CONFIG workdir=$WORKDIR"; date
cd $STORE/baselines/symphony
$PY -u -m symphony --config "$CONFIG" --workdir "$WORKDIR" --use_wandb=False
date
