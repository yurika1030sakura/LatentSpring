#!/bin/bash
#SBATCH -J bgfm-radial-estimators
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
POSITION_CKPT="${3:?radial reference checkpoint required}"
TRAIN_STEPS="${4:-20}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
for ARM in fm exact_value exact_shuffle exact_zero squared_independent product_independent squared_common product_common; do
  EXTRA=()
  WEIGHT=.001
  ESTIMATOR=replica_product
  CONTROL=value
  case "$ARM" in
    fm) WEIGHT=0 ;;
    exact_value) EXTRA+=(--energy-exact-trace) ;;
    exact_shuffle) EXTRA+=(--energy-exact-trace); CONTROL=shuffle ;;
    exact_zero) EXTRA+=(--energy-exact-trace); CONTROL=zero ;;
    squared_independent) ESTIMATOR=squared ;;
    squared_common) ESTIMATOR=squared; EXTRA+=(--common-probes) ;;
    product_common) EXTRA+=(--common-probes) ;;
  esac
  "$FLOWMOL_ENV/bin/python" -u scripts/research/train_clamped_position.py \
   --config configs/sweep/a1_fm_only_s2.yaml --warm-checkpoint "$POSITION_CKPT" \
   --out "$RUN_ROOT/$ARM" --steps "$TRAIN_STEPS" --seed 9012 --terminal-time 1 --lr .0002 \
   --device cpu --lambda-energy "$WEIGHT" --energy-every 1 --energy-steps 64 \
   --energy-solver midpoint --trace-distribution gaussian --perturbation-indices 0 1 \
   --discrete-adjoint-energy --energy-parents 8 --energy-estimator "$ESTIMATOR" \
   --energy-control "$CONTROL" --energy-gradient-diagnostics "${EXTRA[@]}"
done
