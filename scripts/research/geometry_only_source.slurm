#!/bin/bash
#SBATCH -J bgfm-geometry-only-source
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?artifact root required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/prepare_geometry_only_source.py \
  --config configs/sweep/a1_fm_only_s2.yaml --checkpoint "$PROJECT/runs/electronic_fm_global_v1/last.ckpt" \
  --manifest research/evidence/development_panel_v1.json --original-source "$PROJECT/runs/species_breadth_source_full_v2" --out "$RUN_ROOT"
