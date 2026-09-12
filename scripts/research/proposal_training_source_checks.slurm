#!/bin/bash
#SBATCH -J bgfm-training-source-checks
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
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR="$RUN_ROOT/runtime/tmp" XDG_CACHE_HOME="$RUN_ROOT/runtime/cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
PYTHON=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python
"$PYTHON" -m pytest -q -p no:cacheprovider tests/test_proposal_training_selection.py
"$PYTHON" -u scripts/research/audit_transfer_geometry_source.py --project "$PROJECT" \
  --run "$PROJECT/runs/transfer_geometry_source_v1" --out "$RUN_ROOT/transfer_audit_regression.json"
"$PYTHON" - "$PROJECT" "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
project,out=map(Path,sys.argv[1:])
old=json.loads((project/'runs/transfer_geometry_audit_v1/results.json').read_text())
new=json.loads((out/'transfer_audit_regression.json').read_text())
assert old['complete'] and new['complete'] and old['rows']==new['rows']
assert old['protocol_sha256']==new['protocol_sha256'] and old['manifest_sha256']==new['manifest_sha256']
assert new['new_physical_queries']==0
(out/'results.json').write_text(json.dumps(dict(complete=True,training_selection_test_passed=True,
    old_six_composition_source_audit_unchanged=True,new_physical_queries=0),indent=2)+'\n')
print('Full old source audit unchanged; training selection is disjoint and energy blind.')
PY
