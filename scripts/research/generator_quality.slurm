#!/bin/bash
#SBATCH -J lspring-quality
#SBATCH --array=0-7%4
#SBATCH -A ryl_lab
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source}"
PROJECT="${2:?project root}"
RUN_ROOT="${3:?output root}"
KIND="${4:?esen or xtb}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/lspring_quality_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
seed=$((SLURM_ARRAY_TASK_ID / 4))
chunk=$((SLURM_ARRAY_TASK_ID % 4))
cd "$SOURCE_ROOT"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/evaluate_generator_quality.py \
 --project "$PROJECT" --protocol "research/evidence/generator_quality_s${seed}_v1.json" \
 --out "$RUN_ROOT/s${seed}/${KIND}/chunk${chunk}" --kind "$KIND" --chunk "$chunk" \
 --workers "${SLURM_CPUS_PER_TASK}"
