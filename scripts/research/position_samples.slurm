#!/bin/bash
#SBATCH -J bgfm-position-sample
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/sample_position_panel.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint warm /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/abl/a1_fm_only_s2/lightning_logs/version_30674367/checkpoints/midstep-step=30000.ckpt \
 --checkpoint position_1000 /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/position_cfm_v1/last.ckpt \
 --split /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/composition_audit_v2/composition_split.json \
 --out "$RUN_ROOT" --parents 8 --samples 4 --terminal-time .8
