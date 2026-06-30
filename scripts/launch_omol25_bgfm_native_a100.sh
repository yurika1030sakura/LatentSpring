#!/bin/bash
# BGFM-Native training on A100 80GB (production launcher).
#
# Loads configs/native/omol25_4m_bgfm_native_v1.yaml which has
# mol_fm.bgfm.native.enabled: true, so scripts/run_train.py applies
# patch_flowmol_bgfm() (legacy losses) and then patch_bgfm_native()
# (composition-conditioned strain head + energy-coupled vector field
# patch + KL bridge + corrector-in-the-loop).
#
# Usage:
#   sbatch scripts/launch_omol25_bgfm_native_a100.sh                     # full prod run
#   sbatch scripts/launch_omol25_bgfm_native_a100.sh "" --fast           # smoke (10 batches)
#   sbatch scripts/launch_omol25_bgfm_native_a100.sh <resume.ckpt>       # resume

#SBATCH -J omol25_bgfm_native_a100
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_native_v1/train-a100-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_native_v1/train-a100-%j.err
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
