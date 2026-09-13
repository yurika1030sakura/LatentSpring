#!/bin/bash
#SBATCH -J bgfm-edit-bridge-eval
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-1%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
cd "$SOURCE_ROOT"
CASES=(1 2 3 5)
for SLOT in "$SLURM_ARRAY_TASK_ID" "$((SLURM_ARRAY_TASK_ID+2))"; do
  if (( SLOT >= ${#CASES[@]} )); then continue; fi
  INDEX="${CASES[$SLOT]}"
  printf -v PAD '%02d' "$INDEX"
  ARM="$RUN_ROOT/condition_$PAD"
  export TMPDIR="$ARM/runtime/tmp" MPLCONFIGDIR="$ARM/runtime/mpl" XDG_CACHE_HOME="$ARM/runtime/cache" CUDA_CACHE_PATH="$ARM/runtime/cuda"
  mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
  "$STORE/envs/flowmol/bin/python" -u scripts/research/evaluate_edit_bridge.py \
    --phase evaluate --index "$INDEX" --project "$PROJECT" --out "$ARM" \
    --oracle-python "$STORE/envs/omol25/bin/python" \
    --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
done
