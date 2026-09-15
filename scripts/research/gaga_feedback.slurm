#!/bin/bash
#SBATCH -J lspring-gaga-feedback
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lspring_feedback_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_feedback_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE"
seed="$SLURM_ARRAY_TASK_ID"
for context in distance tree; do
 "$FLOWMOL_ENV/bin/python" -u scripts/research/run_gaga_feedback.py \
  --project "$PROJECT" --protocol "$SOURCE/research/evidence/gaga_feedback_${context}_s${seed}_v1.json" \
  --out "$RUN_ROOT/s${seed}/${context}" --seed "$seed" --stage train
done
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/run_gaga_feedback.py \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/gaga_feedback_distance_s${seed}_v1.json" \
 --out "$RUN_ROOT/s${seed}/baselines" --seed "$seed" --stage baselines
