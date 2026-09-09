#!/bin/bash
#SBATCH -J bgfm-cnf-panel
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source snapshot required}"
RUN_ROOT="${2:?output directory required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/checkpoint_panel.py \
  --config configs/sweep/a1_fm_only_s2.yaml \
  --checkpoint /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/abl/a1_fm_only_s2/lightning_logs/version_30674367/checkpoints/midstep-step=30000.ckpt \
  --shard /n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed/perturbation_val_n10000_s0.pt \
  --parents 8 --replicas 8 --max-atoms 12 --out "$RUN_ROOT"
