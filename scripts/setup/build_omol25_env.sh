#!/bin/bash
# Build the omol25 conda env (torch + fairchem-core + ase) on UMass Unity.
# Used for: reading OMol25 .aselmdb (preprocessing) and the gated UMA/OMol25
# neural-potential oracle (FAIRChemCalculator). Env lives in scratch.
set -eo pipefail
SCRATCH=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
PREFIX=$SCRATCH/envs/omol25
export CONDA_PKGS_DIRS=$SCRATCH/.conda_pkgs
mkdir -p "$CONDA_PKGS_DIRS"

echo "=== [omol25-env] start $(date) on $(hostname) ==="
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -x "$PREFIX/bin/python" ]; then
  echo "env already exists at $PREFIX; skipping create"
else
  conda create --prefix "$PREFIX" python=3.11 -y
fi

echo "=== pip install fairchem-core + ase (pulls CUDA torch from PyPI) ==="
"$PREFIX/bin/pip" install --upgrade pip
"$PREFIX/bin/pip" install fairchem-core ase

echo "=== verify ==="
"$PREFIX/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
import ase; print("ase", ase.__version__)
import fairchem.core as fc
print("fairchem.core", getattr(fc, "__version__", "?"))
from fairchem.core.datasets import AseDBDataset
print("AseDBDataset import OK (preprocessing reader)")
try:
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    print("FAIRChemCalculator + load_predict_unit import OK (oracle, needs gated ckpt + HF login)")
except Exception as e:
    print("oracle import note:", repr(e))
PY
echo "=== [omol25-env] DONE $(date) ==="
