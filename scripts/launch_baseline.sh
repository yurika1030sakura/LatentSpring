#!/bin/bash
# Train UNPATCHED FlowMol3 baseline on QM9 for ICLR paper comparison.
#
# This is the literal FlowMol3 99.9%-validity run (no constraint patches,
# no cfm_mol involvement). Purpose: produce our own FlowMol3 checkpoint
# on IDENTICAL data split + featurisation so the in-distribution + OOD
# comparison against cfm_mol is apples-to-apples (authors' reported
# numbers use slightly different seeds / atom map / featurisation).
#
# Usage:
#   sbatch scripts/launch_baseline.sh              # QM9
#   sbatch scripts/launch_baseline.sh geom         # GEOM-Drugs
#
#SBATCH -J cfm_baseline
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/baseline-%j.out

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
FLOWMOL=$PROJ/baselines/flowmol3
cd "$PROJ"

DATASET="${1:-qm9}"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

echo "host: $(hostname)"
echo "gpu:"
nvidia-smi -L
echo ""

if [ "$DATASET" = "qm9" ]; then
    CONFIG="$PROJ/configs/qm9_cfm.yaml"
elif [ "$DATASET" = "geom" ]; then
    CONFIG="$FLOWMOL/configs/flowmol3.yml"
else
    echo "unknown dataset: $DATASET (expected qm9 or geom)"
    exit 2
fi

echo "training UNPATCHED FlowMol3 baseline on $DATASET (config: $CONFIG)"
cd "$FLOWMOL"
python train.py --config "$CONFIG"
