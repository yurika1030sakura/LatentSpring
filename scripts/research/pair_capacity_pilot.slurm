#!/bin/bash
#SBATCH -J bgfm-pair-capacity
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/pair_capacity_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/pair_capacity_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
SEED=$((SLURM_ARRAY_TASK_ID / 2))
if ((SLURM_ARRAY_TASK_ID % 2)); then KIND=pair_affine; else KIND=pair; fi
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/train_particle_benchmark.py \
 --system dw4 --kind "$KIND" --sweeps 4 --steps 1000 --batch 128 --scale 1.6 \
 --pair-curvature-bound .75 --lr .001 --seed "$SEED" --eval-samples 8192 \
 --device cpu --out "$RUN_ROOT/${KIND}_s${SEED}.json"
