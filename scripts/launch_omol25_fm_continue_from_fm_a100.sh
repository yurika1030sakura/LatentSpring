#!/bin/bash
# Same-budget FM continuation control on OMol25 4M.
#
# This is the control for the clean FM -> BGFM fine-tune:
#   Start from the same FM checkpoint, use the same low LR and walltime/checkpoint
#   cadence, but do NOT enable BGFM. If this control does not improve the physics
#   metrics like BGFM, the paper can attribute gains to BGFM rather than extra
#   training alone.
#
# Usage:
#   sbatch scripts/launch_omol25_fm_continue_from_fm_a100.sh [fm_ckpt]

#SBATCH -J omol25_fm_continue
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_fm_continue_from_fm/train-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_fm_continue_from_fm/train-%j.err
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG="$PROJ/configs/omol25_4m_cfm.yaml"
RESUME_CKPT="${1:-$PROJ/runs/omol25_4m_cfm/lightning_logs/version_8753404/checkpoints/epoch=0-step=292658.ckpt}"
OUT_DIR="$PROJ/runs/omol25_4m_cfm_continue_from_fm"

mkdir -p "$PROJ/runs/omol25_fm_continue_from_fm" "$OUT_DIR"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
nvidia-smi -L
echo "config: $CONFIG"
echo "resume FM weights: $RESUME_CKPT"
echo "output_dir: $OUT_DIR"
echo "variant: FM same-budget continuation control, no BGFM"

python -u scripts/run_train.py \
  --config "$CONFIG" \
  --resume_from "$RESUME_CKPT" \
  --override_output_dir "$OUT_DIR" \
  --override_base_lr 3.0e-5 \
  --override_max_epochs 4
