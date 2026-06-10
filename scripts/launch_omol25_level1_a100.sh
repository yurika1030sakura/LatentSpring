#!/bin/bash
# Level-1 training (bond-free, no BGFM) on A100 80GB. Backup to H100
# version (7202517). FM-only baseline for BGFM ablation matrix.

#SBATCH -J omol25_l1_a100
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_train/train-a100-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_train/train-a100-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_train"
CONFIG="${1:-$PROJ/configs/omol25_4m_cfm.yaml}"
RESUME_CKPT="${2:-}"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "host: $(hostname)"; nvidia-smi -L; echo "config: $CONFIG"

RESUME_ARG=""
[ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ] && RESUME_ARG="--resume_ckpt_path $RESUME_CKPT"

export PYTHONUNBUFFERED=1
python -u scripts/run_train.py --config "$CONFIG" $RESUME_ARG
