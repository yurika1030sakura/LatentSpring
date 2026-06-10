#!/bin/bash
# Plan B Level-2: Boltzmann-Guided Flow Matching on OMol25 4M.
# Distinct from Level-1 (launch_omol25_train_h200.sh) -- this one uses the
# BGFM config with force-consistency auxiliary loss enabled.
#
# Expected: ~2-3x slower per step than Level-1 (extra forward at t_eval
# for score computation). 10 epochs ~ 5-8 days on H100.
#
# Usage: sbatch scripts/launch_omol25_bgfm_h100.sh [config] [resume_ckpt]

#SBATCH -J omol25_bgfm
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_bgfm/train-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_bgfm/train-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_bgfm"

CONFIG="${1:-$PROJ/configs/omol25_4m_bgfm.yaml}"
RESUME_CKPT="${2:-}"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "host: $(hostname)"
echo "gpu:"; nvidia-smi -L
echo "config: $CONFIG"
echo "resume: ${RESUME_CKPT:-none}"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    RESUME_ARG="--resume_ckpt_path $RESUME_CKPT"
fi

export PYTHONUNBUFFERED=1

python -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG
