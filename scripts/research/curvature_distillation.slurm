#!/bin/bash
#SBATCH -J lspring-curvature-fm
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source}"
PROJECT="${2:?project root}"
RUN_ROOT="${3:?output root}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/lspring_curvature_fm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -u scripts/research/run_curvature_escort.py \
 --project "$PROJECT" --protocol "research/evidence/curvature_full_teacher_s${SLURM_ARRAY_TASK_ID}_v1.json" \
 --out "$RUN_ROOT/teacher/s${SLURM_ARRAY_TASK_ID}"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/run_curvature_distillation.py \
 --project "$PROJECT" --protocol "research/evidence/curvature_distillation_s${SLURM_ARRAY_TASK_ID}_v1.json" \
 --out "$RUN_ROOT/student/s${SLURM_ARRAY_TASK_ID}"
