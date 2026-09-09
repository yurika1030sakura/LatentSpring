#!/bin/bash
#SBATCH -J bgfm-electronic-assess
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
RESEARCH_ROOT=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PATH="$FLOWMOL_ENV/bin:$PATH" PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_electronic_assess_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_electronic_assess_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$FLOWMOL_ENV/bin/python" -c 'import json;from pathlib import Path;root=Path("/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs");assert all(json.loads((root/name/"validation.json").read_text())["complete"] for name in ["electronic_fm_global_v1","electronic_fm_legacy_v1"])'
"$FLOWMOL_ENV/bin/python" -u scripts/research/sample_position_panel.py \
 --config configs/sweep/a1_fm_only_s2.yaml \
 --checkpoint legacy_continued "$RESEARCH_ROOT/runs/electronic_fm_legacy_v1/last.ckpt" \
 --checkpoint global_electronic "$RESEARCH_ROOT/runs/electronic_fm_global_v1/last.ckpt" \
 --split "$RESEARCH_ROOT/runs/composition_audit_v2/composition_split.json" \
 --out "$RUN_ROOT/sampling" --terminal-time 1 --device cpu
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/eval_position_xtb.py \
 --samples "$RUN_ROOT/sampling/samples.json" --out "$RUN_ROOT/xtb" --workers 2
