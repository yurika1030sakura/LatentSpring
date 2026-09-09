#!/bin/bash
#SBATCH -J bgfm-cnf-smoke
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source directory required}"
RUN_ROOT="${2:?run output directory required}"
CONFIG_PATH="${3:-configs/research/runtime_smoke.yaml}"
TRAIN_SEED="${4:-9001}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
nvidia-smi -L
exec "$FLOWMOL_ENV/bin/python" -u scripts/run_train.py \
  --config "$CONFIG_PATH" --seed "$TRAIN_SEED" \
  --override_output_dir "$RUN_ROOT" --no_auto_resume \
  --resume_from /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/abl/a1_fm_only_s2/lightning_logs/version_30674367/checkpoints/midstep-step=30000.ckpt
