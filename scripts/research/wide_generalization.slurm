#!/bin/bash
#SBATCH -J lspring-wide
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
BASE_SOURCE="${1:?immutable base source}"
ENTRY_SOURCE="${2:?immutable current protocols}"
RUN_ROOT="${3:?run root}"
PROJECT="${4:?project data root}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$BASE_SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lspring_wide_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_wide_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$BASE_SOURCE"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/evaluate_source_sc_confirmation.py \
 --project "$PROJECT" --protocol "$ENTRY_SOURCE/research/evidence/wide_generalization_s${SLURM_ARRAY_TASK_ID}_v1.json" \
 --out "$RUN_ROOT/s${SLURM_ARRAY_TASK_ID}/evaluation"
