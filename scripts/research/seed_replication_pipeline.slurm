#!/bin/bash
#SBATCH -J lspring-seed-pipeline
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?source}"
PROJECT="${2:?project}"
RUN="${3:?run}"
OFFSET="${4:-2}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_seed_pipeline_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME"
srun --export=ALL --gpu-bind=single:1 --kill-on-bad-exit=0 --output="$PROJECT/logs/lspring-seed-worker-%j-%t.out" \
 --error="$PROJECT/logs/lspring-seed-worker-%j-%t.err" \
 bash "$SOURCE/scripts/research/seed_replication_worker.sh" "$SOURCE" "$PROJECT" "$RUN" "$OFFSET"
