#!/bin/bash
#SBATCH -J cfm_qm9
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/slurm-%j.out

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
cd "$PROJ"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

echo "host: $(hostname)"
echo "gpu:"
nvidia-smi -L
echo ""

# Use FlowMol3's own train.py entry point; we do the patching inside the
# config-loaded model via scripts/run_train.py that wraps cfm_mol.flow_model.
# Optional: pass a different config as $1 (e.g. configs/geom_cfm.yaml).
CONFIG="${1:-$PROJ/configs/qm9_cfm.yaml}"
echo "training with config: $CONFIG"
python "$PROJ/scripts/run_train.py" --config "$CONFIG"
