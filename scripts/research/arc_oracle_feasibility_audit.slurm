#!/bin/bash
#SBATCH -J bgfm-arc-oracle-audit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
INPUT="${4:?input run required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$RUN_ROOT/runtime/tmp" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
for INDEX in 1 2 3 5; do
  printf -v PAD '%02d' "$INDEX"
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/evaluate_arc_oracle_feasibility.py \
    --phase audit --index "$INDEX" --project "$PROJECT" --run "$INPUT/condition_$PAD" --out "$RUN_ROOT/condition_$PAD/results.json"
done
