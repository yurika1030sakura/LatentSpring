#!/bin/bash
#SBATCH -J bgfm-radial-reference
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
TRAIN_STEPS="${3:-10000}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PATH="$FLOWMOL_ENV/bin:$PATH"
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="${SLURM_TMPDIR:-/tmp}/bgfm_${SLURM_JOB_ID}"
export MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -u scripts/research/train_clamped_position.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --warm-checkpoint /n/holylabs/woo_lab/Lab/yulili/bgfm/runs/abl/a1_fm_only_s2/lightning_logs/version_30674367/checkpoints/midstep-step=30000.ckpt \
 --out "$RUN_ROOT/training" --steps "$TRAIN_STEPS" --seed 9004 --terminal-time 1 --lr .001 \
 --device cpu --position-backbone radial_reference --position-parameterization displacement
"$FLOWMOL_ENV/bin/python" -u scripts/research/sample_position_panel.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint radial_reference "$RUN_ROOT/training/last.ckpt" \
 --out "$RUN_ROOT/assessment/sampling" --terminal-time 1 --device cpu \
 --split /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/composition_audit_v2/composition_split.json
"$FLOWMOL_ENV/bin/python" -u scripts/research/eval_position_xtb.py \
 --samples "$RUN_ROOT/assessment/sampling/samples.json" \
 --out "$RUN_ROOT/assessment/xtb" --workers 2
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/checkpoint_panel.py \
 --config configs/sweep/a1_fm_only_s2.yaml --checkpoint "$RUN_ROOT/training/last.ckpt" \
 --shard /n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed/perturbation_val_n10000_s0.pt \
 --parents 8 --replicas 8 --max-atoms 12 --out "$RUN_ROOT/density" --device cpu \
 --terminal-time 1 --exact-parent-count 8 --exact-steps 16 32 64 --steps 16 32 64 --perturbation-indices 0 1 \
 --composition-split /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/composition_audit_v2/composition_split.json
