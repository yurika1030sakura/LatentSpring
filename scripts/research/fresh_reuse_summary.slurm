#!/bin/bash
#SBATCH -J bgfm-fresh-reuse-summary
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
EVALUATION_RUN="${4:?evaluation directory required}"
AUDIT_RUN="${5:?audit directory required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/summarize_fresh_reuse.py \
  --project "$PROJECT" --run "$EVALUATION_RUN" --audits "$AUDIT_RUN" --out "$RUN_ROOT/results.json"
"$STORE/envs/flowmol/bin/python" -u scripts/research/plot_fresh_reuse.py \
  --summary "$RUN_ROOT/results.json" --out "$RUN_ROOT/figures"
