#!/bin/bash
#SBATCH -J bgfm-backward-refit
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_REFIT_STEPS="${3:-500}"
BGFM_TRAIN_PATHS="${4:-2048}"
BGFM_EVAL_PATHS="${5:-256}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_refit_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_refit_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/refit_backward.py \
 --source-run /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/work300_annealed_joint_5846_1500_v1 \
 --out "$RUN_ROOT" --steps "$BGFM_REFIT_STEPS" --train-paths "$BGFM_TRAIN_PATHS" --eval-paths "$BGFM_EVAL_PATHS"
