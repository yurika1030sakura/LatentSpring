#!/bin/bash
#SBATCH -J lspring-seed-eval
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-4%3
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE="${1:?source}"
PROJECT="${2:?project}"
RUN="${3:?run}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_seed_eval_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME"
cd "$SOURCE"
exec "$FLOW_ENV/bin/python" -u -m scripts.research.run_seed_replication \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/seed_replication_v1.json" \
 --fit "$SLURM_ARRAY_TASK_ID" --out "$RUN/evaluation/s${SLURM_ARRAY_TASK_ID}"
