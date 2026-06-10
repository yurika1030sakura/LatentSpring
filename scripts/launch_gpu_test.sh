#!/bin/bash
# Quick GPU-pipeline validation on gpu_test partition (short walltime).
#
# Runs 1 epoch, 10 train batches, 1 val batch on QM9 with full patch_flowmol.
# Succeeds ≈ ready to commit real training compute. Fails ≈ bugs to fix
# before spending a multi-GPU-day real run.
#
# Expected ≤ 15 min on A100. gpu_test walltime cap is typically 1h so
# headroom is fine.

#SBATCH -J cfm_gputest
#SBATCH -p gpu_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/gputest-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/gputest-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
cd "$PROJ"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

echo "host: $(hostname)"
echo "start: $(date)"
echo "gpu:"
nvidia-smi -L
echo ""

# QM9 config, full patch, fast dev-run.
python scripts/run_train.py \
    --config "$PROJ/configs/qm9_cfm.yaml" \
    --fast

echo ""
echo "done: $(date)"
