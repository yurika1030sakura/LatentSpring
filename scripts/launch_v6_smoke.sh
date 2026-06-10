#!/bin/bash
#SBATCH -J v6_smoke
#SBATCH -p gpu_test
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy_v6_mini/smoke-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_4m_bgfm_energy_v6_mini/smoke-%j.err
set -e
PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_4m_bgfm_energy_v6_mini"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host: $(hostname)"; nvidia-smi -L
python -u scripts/run_train.py --config configs/omol25_4m_bgfm_energy_v6_mini.yaml --fast
echo "=== V6 SMOKE DONE (exit $?) ==="
