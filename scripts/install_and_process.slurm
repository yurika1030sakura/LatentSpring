#!/bin/bash
#SBATCH -J cfm_install
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/install-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/install-%j.err

set -e
PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs"

echo "host: $(hostname)"
echo "start: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Step 1: create conda env (with always_copy to dodge NFS hardlink issues)
# ---------------------------------------------------------------------------
echo "=== Step 1: env install ==="
cd "$PROJ"
rm -rf envs/flowmol 2>/dev/null || true
bash scripts/setup_env.sh
echo "env install done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Step 2: verify env
# ---------------------------------------------------------------------------
echo "=== Step 2: verify env ==="
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
python -c "
import ctypes
import torch, dgl, pytorch_lightning, pandas, yaml, tqdm
from rdkit import Chem
from flowmol.data_processing.geom import MoleculeFeaturizer
import cfm_mol
print('all imports OK')
print('torch:', torch.__version__)
print('dgl:', dgl.__version__)
"
echo "verify done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Step 3: process QM9
# ---------------------------------------------------------------------------
echo "=== Step 3: process QM9 ==="
cd "$PROJ/baselines/flowmol3"
python process_qm9.py \
    --config "$PROJ/configs/qm9_kekulized.yaml" \
    --n_cpus 6
echo "qm9 processing done: $(date)"
echo ""

# ---------------------------------------------------------------------------
# Step 4: run cfm_mol sanity tests inside the flowmol env too
# ---------------------------------------------------------------------------
echo "=== Step 4: cfm_mol sanity tests ==="
cd "$PROJ"
python -m cfm_mol.test_fibre
echo "sanity tests done: $(date)"
echo ""

echo "ALL DONE: $(date)"
echo "Next: bash scripts/launch_train.sh  # when you want to start training"
