#!/bin/bash
#SBATCH -J bgfm-matched-control
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
REPLICA="${3:-0}"
KIND="${4:-convex}"
MODE="${5:-production}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matched_control_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/matched_control_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
SOURCE_POOL="$LAB_RUNS/species_breadth_source_full_v2"
if [ "$MODE" = smoke ]; then SOURCE_POOL="$LAB_RUNS/species_breadth_source_smoke_v2"; fi
ARGS=(--source-root "$SOURCE_POOL" --condition-index 0 --out "$RUN_ROOT"
      --kind "$KIND" --replica "$REPLICA" --parity-target --device cpu --oracle-device cuda --oracle-batch-size 32)
if [ "$MODE" = smoke ]; then ARGS+=(--engineering-smoke --steps 2 --eval-count 16);
elif [ "$MODE" != production ]; then exit 2; fi
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/train_broad_entropy_adapter.py "${ARGS[@]}"
