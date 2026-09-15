#!/bin/bash
#SBATCH -J lspring-source-repeat
#SBATCH --array=2-4%3
#SBATCH -A ryl_lab
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE="${1:?immutable source}"
PROJECT="${2:?project data root}"
RUN_ROOT="${3:?run root}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lspring_repeat_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_repeat_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/run_source_replication.py \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/source_replication_s${SLURM_ARRAY_TASK_ID}_v1.json" \
 --out "$RUN_ROOT/s${SLURM_ARRAY_TASK_ID}"
