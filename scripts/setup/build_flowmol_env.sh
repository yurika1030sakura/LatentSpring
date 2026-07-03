#!/bin/bash
# Build the flowmol conda env (torch 2.2/cu121 + DGL + Lightning + FlowMol3 + cfm_mol)
# on UMass Unity. Env lives in scratch (off the small $HOME quota).
set -eo pipefail
REPO=/home/renhaozhang_umass_edu/bgfm
SCRATCH=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PREFIX=$SCRATCH/envs/flowmol
export CONDA_PKGS_DIRS=$SCRATCH/.conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"

echo "=== [flowmol-env] start $(date) on $(hostname) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -x "$PREFIX/bin/python" ]; then
  echo "env already exists at $PREFIX; skipping create"
else
  echo "=== creating env from environment.yml (this is the slow part) ==="
  conda env create --prefix "$PREFIX" -f "$REPO/baselines/flowmol3/environment.yml"
fi

echo "=== pip install -e FlowMol3 ==="
"$PREFIX/bin/pip" install -e "$REPO/baselines/flowmol3" --no-deps
echo "=== pip install -e cfm_mol ==="
"$PREFIX/bin/pip" install -e "$REPO" --no-deps

echo "=== verify imports ==="
"$PREFIX/bin/python" - <<'PY'
import torch, dgl, pytorch_lightning
import rdkit
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("lightning", pytorch_lightning.__version__)
try:
    import flowmol
    print("flowmol import OK")
except Exception as e:
    print("flowmol import FAILED:", repr(e))
try:
    import cfm_mol
    print("cfm_mol import OK")
except Exception as e:
    print("cfm_mol import FAILED:", repr(e))
PY
echo "=== [flowmol-env] DONE $(date) ==="
