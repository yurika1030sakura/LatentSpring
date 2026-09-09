#!/bin/bash
#SBATCH -J bgfm-energy-pilot
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
POSITION_CKPT="${3:?position checkpoint required}"
TERMINAL_TIME="${4:?terminal time required}"
ESTIMATOR="${5:-replica_product}"
CONTROL="${6:-value}"
TRAIN_SEED="${7:-9012}"
TRAIN_STEPS="${8:-2}"
ENERGY_PARENTS="${9:-8}"
LAMBDA_ENERGY="${10:-.001}"
ODE_STEPS="${11:-128}"
ODE_SOLVER="${12:-midpoint}"
TRACE_DISTRIBUTION="${13:-rademacher}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/train_clamped_position.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --warm-checkpoint "$POSITION_CKPT" \
 --out "$RUN_ROOT" --steps "$TRAIN_STEPS" --seed "$TRAIN_SEED" --terminal-time "$TERMINAL_TIME" --lr 2e-5 \
 --lambda-energy "$LAMBDA_ENERGY" --energy-every 1 --energy-steps "$ODE_STEPS" \
 --energy-solver "$ODE_SOLVER" --trace-distribution "$TRACE_DISTRIBUTION" \
 --perturbation-indices 0 1 --common-probes --discrete-adjoint-energy \
 --energy-parents "$ENERGY_PARENTS" --energy-estimator "$ESTIMATOR" --energy-control "$CONTROL" \
 --energy-gradient-diagnostics
