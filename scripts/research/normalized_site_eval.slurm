#!/bin/bash
#SBATCH -J bgfm-normalized-site-eval
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
TABLE="${3:?table required}"
MODELS="${4:?models required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/cache" CUDA_CACHE_PATH="$RUN_ROOT/runtime/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$SOURCE_ROOT"
for METHOD in site initial_mixture learned_vector learned_mixture; do
  for REPLICA in 0 1; do
    "$STORE/envs/flowmol/bin/python" -u scripts/research/evaluate_normalized_site.py \
      --table "$TABLE" --models "$MODELS" --out "$RUN_ROOT/${METHOD}_s${REPLICA}" \
      --method "$METHOD" --replica "$REPLICA" --oracle-python "$STORE/envs/omol25/bin/python" \
      --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
  done
done
