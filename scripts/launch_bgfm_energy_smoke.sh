#!/bin/bash
# Smoke test for full BGFM (force + energy / lambda_2 > 0). Runs --fast
# (1 epoch, 10 train batches) to verify the FFJORD energy term does not
# NaN / OOM before committing to multi-day training. With
# energy_every_k_steps=4 the energy loss fires at batches 0,4,8.

#SBATCH -J bgfm_energy_smoke
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/smoke-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy/smoke-%j.err

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
python -u scripts/run_train.py --config configs/omol25_4m_bgfm_energy.yaml --fast
echo "=== SMOKE DONE (exit $?) ==="
