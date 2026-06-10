#!/bin/bash
# BGFM training on A100 80GB (backup to H100 version). Usually starts
# faster because the gpu partition is dedicated (not preemptible).
# Estimated slowdown: 1.5-2x vs H100 due to lower FLOPS + memory bandwidth.
# Memory (80GB) same as H100, so batch size unchanged.

#SBATCH -J omol25_bgfm_energy_h200
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/train-a100-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/train-a100-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_bgfm"
CONFIG="${1:-$PROJ/configs/omol25_4m_bgfm_energy.yaml}"
RESUME_CKPT="${2:-}"
SEED="${3:-42}"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host: $(hostname)"; nvidia-smi -L; echo "config: $CONFIG"

RESUME_ARG=""
# weights-only fine-tune: load model state but reinit optimizer + epoch counter.
# This is right for adding the lambda_2 * L_energy term to a force-only-trained
# checkpoint -- the new loss landscape needs fresh optimizer momentum, not the
# stale state from a different loss surface.
[ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ] && RESUME_ARG="--resume_from $RESUME_CKPT"

export PYTHONUNBUFFERED=1
python -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG --seed "$SEED"
