#!/bin/bash
#SBATCH -J ls-geometry-recovery
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%3
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE="${1:?source}"; PROJECT="${2:?project}"; RUN="${3:?run}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/ls_geometry_recovery_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME"
fit=$((SLURM_ARRAY_TASK_ID / 3))
variants=(replay recovery recovery_local)
variant="${variants[$((SLURM_ARRAY_TASK_ID % 3))]}"
cd "$SOURCE"
exec "$FLOW_ENV/bin/python" -s -u -m scripts.research.run_geometry_recovery \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/geometry_recovery_v1.json" \
 --fit "$fit" --variant "$variant" --out "$RUN/s${fit}/${variant}"
