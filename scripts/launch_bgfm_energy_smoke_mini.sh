#!/bin/bash
# Mini smoke for the fixed per-mol energy loss. Fits in 20GB MIG slice on
# gpu_test partition, which has open backfill slots while gpu_requeue is
# starved by the boltz queue.
#
# What this verifies:
#   - PerturbationLoader loads the shards + builds DGL graphs correctly
#   - log_density_via_flow returns finite values with the small graph
#   - energy_consistency_loss_per_mol produces O(1) - O(100) magnitude
#     (the previous broken cross-mol form gave 1e9)
#   - Gradients flow without NaN
#
# Does NOT verify (need full-size smoke for these):
#   - max_atoms=200 / batch_size=16 fits the energy term on A100-80GB
#   - Real training-scale gradient noise

#SBATCH -J bgfm_energy_smoke_mini
#SBATCH -p gpu_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/smoke_mini-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/smoke_mini-%j.err

set -e
PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_4m_bgfm_energy"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "host: $(hostname)"; nvidia-smi -L
python -u scripts/run_train.py --config configs/omol25_4m_bgfm_energy_mini.yaml --fast
echo "=== MINI SMOKE DONE (exit $?) ==="
