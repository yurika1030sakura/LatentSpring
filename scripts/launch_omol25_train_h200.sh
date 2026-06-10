#!/bin/bash
# Plan B: Train OMol25-4M flow-matching generator on a single H100 (80GB).
# Expected: ~6-11 hours/epoch on H100 bf16; 10 epochs ~ 3-5 days. Config's
# save-last + save-top-3 checkpointing lets us resume across walltime or
# preemption boundaries (gpu_requeue partition is preemptible).
#
# Usage:
#   sbatch scripts/launch_omol25_train_h200.sh [config_path] [resume_ckpt]

#SBATCH -J omol25_train
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_train/train-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_train/train-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_train"

CONFIG="${1:-$PROJ/configs/omol25_4m_cfm.yaml}"
RESUME_CKPT="${2:-}"        # optional: path to last.ckpt for walltime-restart

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "host: $(hostname)"
echo "gpu:"; nvidia-smi -L
echo ""
echo "config: $CONFIG"
echo "resume: ${RESUME_CKPT:-none}"

RESUME_ARG=""
if [ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ]; then
    RESUME_ARG="--resume_ckpt_path $RESUME_CKPT"
fi

export PYTHONUNBUFFERED=1
python scripts/run_train.py --config "$CONFIG" $RESUME_ARG
