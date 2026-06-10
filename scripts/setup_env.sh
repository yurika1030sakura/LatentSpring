#!/bin/bash
# Set up conda env for FlowMol3 + our cfm_mol code.
# One-time setup. Takes ~20 minutes (conda resolve + dgl CUDA wheels).
set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
FLOWMOL=$PROJ/baselines/flowmol3
# Env lives in netscratch (holylabs NFS produces stale-file-handle errors during
# conda install's many-small-file extraction); symlink at $PROJ/envs/flowmol.
ENV_NETSCRATCH=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol_envs/flowmol
ENV_PREFIX=$PROJ/envs/flowmol

# Pkg cache on netscratch (was on holylabs; holylabs NFS throws
# "cannot remove all: No such file or directory" during pytorch extraction
# because NFS caches stale directory entries that conda's cleanup trips on).
export CONDA_PKGS_DIRS=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol_pkgs
export CONDARC=$PROJ/.condarc_local/.condarc
mkdir -p "$CONDA_PKGS_DIRS"
mkdir -p "$(dirname $ENV_NETSCRATCH)"
mkdir -p "$(dirname $ENV_PREFIX)"

# Clean any half-extracted dirs from a prior failed run on the holylabs cache.
# If we re-use $PROJ/.conda_pkgs later for anything, zombie pytorch entries
# under it confuse mamba. Harmless if missing.
rm -rf "$PROJ/.conda_pkgs/pytorch-2.2.0-py3.10_cuda12.1_cudnn8.9.2_0" 2>/dev/null || true

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

if [ -d "$ENV_PREFIX" ] && [ -f "$ENV_PREFIX/bin/python" ]; then
    echo "flowmol env already exists at $ENV_PREFIX; skipping create."
else
    echo "creating flowmol env at $ENV_PREFIX (via mamba) ..."
    mamba env create --prefix "$ENV_PREFIX" -f "$FLOWMOL/environment.yml"
fi

conda activate "$ENV_PREFIX"

echo "installing flowmol package in editable mode ..."
pip install -e "$FLOWMOL"

echo "installing cfm_mol in editable mode ..."
pip install -e "$PROJ"

echo ""
echo "DONE. Verify with:"
echo "  conda activate $ENV_PREFIX"
echo "  python -c \"import dgl, pytorch_lightning, flowmol, cfm_mol; print('ok')\""
