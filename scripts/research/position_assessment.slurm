#!/bin/bash
#SBATCH -J bgfm-position-assess
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
POSITION_CKPT="${3:-/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/position_cfm_10000_s9004_v2/last.ckpt}"
ARM_NAME="${4:-position_10000}"

FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -u scripts/research/sample_position_panel.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint "$ARM_NAME" "$POSITION_CKPT" \
 --split /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/composition_audit_v2/composition_split.json \
 --out "$RUN_ROOT/sampling" --parents 8 --samples 4 --terminal-time .8
export PATH="$FLOWMOL_ENV/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/eval_position_xtb.py \
 --samples "$RUN_ROOT/sampling/samples.json" --out "$RUN_ROOT/xtb" --workers 4
