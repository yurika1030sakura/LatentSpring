#!/bin/bash
# Apply the BGFM FlowMol3 patch (wires DFT forces into g.ndata['force_1_true'] so the
# physics loss actually trains; also the dataset_name whitelist + atom_types one-hot).
# WITHOUT this, the BGFM hook silently skips all physics losses and trains plain FM.
# Usage: bash scripts/flowmol3_patch/apply.sh /path/to/flowmol3  (default: pip location)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FM="${1:-$(python -c 'import flowmol,os;print(os.path.dirname(flowmol.__file__))' 2>/dev/null)}"
[ -z "$FM" ] && { echo "could not locate flowmol; pass its dir as arg 1"; exit 1; }
TGT="$FM/data_processing/dataset.py"
[ -f "$TGT" ] || TGT="$FM/flowmol/data_processing/dataset.py"
echo "target: $TGT"
cp "$TGT" "$TGT.pre_bgfm.bak"
cp "$HERE/dataset.py" "$TGT"
echo "patched dataset.py (backup at $TGT.pre_bgfm.bak)."
grep -q "force_1_true" "$TGT" && echo "OK: force_1_true wiring present." || echo "WARN: force_1_true not found!"
