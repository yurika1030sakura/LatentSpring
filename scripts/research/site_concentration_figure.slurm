#!/bin/bash
#SBATCH -J bgfm-site-concentration-figure
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/plot_site_concentration_controls.py \
  --project "$PROJECT" --out "$RUN_ROOT"
