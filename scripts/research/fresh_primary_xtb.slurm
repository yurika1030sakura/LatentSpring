#!/bin/bash
#SBATCH -J bgfm-fresh-primary-xtb
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/export_fresh_primary_endpoints.py \
  --project "$PROJECT" --out "$RUN_ROOT/endpoints.json"
"$STORE/envs/flowmol/bin/python" -u scripts/research/evaluate_fresh_primary_xtb.py \
  --project "$PROJECT" --tasks "$RUN_ROOT/endpoints.json" --binary "$STORE/envs/flowmol/bin/xtb" --out "$RUN_ROOT/xtb" --workers 8
"$STORE/envs/flowmol/bin/python" -u scripts/research/audit_fresh_primary_xtb.py \
  --project "$PROJECT" --tasks "$RUN_ROOT/endpoints.json" --run "$RUN_ROOT/xtb" --out "$RUN_ROOT/audit.json"
