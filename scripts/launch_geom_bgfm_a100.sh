#!/bin/bash
# Cell B: BGFM on GEOM data with OMol25-annotated forces/energies.
# Isolates the *method* contribution from training-data choice.

#SBATCH -J geom_bgfm
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 3-00:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/geom_bgfm/train-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/geom_bgfm/train-%j.err
#SBATCH --signal=SIGUSR1@120

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/geom_bgfm"
CONFIG="${1:-$PROJ/configs/geom_bgfm.yaml}"
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
