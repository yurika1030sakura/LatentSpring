#!/bin/bash
# BGFM-Native training on H100 80GB via gpu_requeue (preemptible but
# typically dequeues faster than the dedicated gpu partition).
#
# Companion to launch_omol25_bgfm_native_a100.sh: same config, different
# hardware request. Use --requeue with SIGUSR1@120 so Lightning checkpoints
# at preemption time and the next run resumes from --resume_ckpt_path.
#
# Usage:
#   sbatch scripts/launch_omol25_bgfm_native_h100.sh                     # production
#   sbatch scripts/launch_omol25_bgfm_native_h100.sh "" --fast           # smoke
#   sbatch scripts/launch_omol25_bgfm_native_h100.sh <resume.ckpt>       # resume

#SBATCH -J omol25_bgfm_native_h100
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_native_v1/train-h100-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_native_v1/train-h100-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CFG="$PROJ/configs/native/omol25_4m_bgfm_native_v1.yaml"
RESUME_CKPT="${1:-}"
EXTRA_ARGS="${2:-}"

mkdir -p "$PROJ/runs/omol25_4m_bgfm_native_v1"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "host:    $(hostname)"
echo "config:  $CFG"
echo "resume:  $RESUME_CKPT"
echo "extra:   $EXTRA_ARGS"
nvidia-smi -L

RESUME_ARG=""
[ -n "$RESUME_CKPT" ] && [ -f "$RESUME_CKPT" ] && RESUME_ARG="--resume_ckpt_path $RESUME_CKPT"

export PYTHONUNBUFFERED=1
python -u scripts/run_train.py --config "$CFG" $RESUME_ARG $EXTRA_ARGS
