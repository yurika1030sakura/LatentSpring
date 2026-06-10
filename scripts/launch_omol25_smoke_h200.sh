#!/bin/bash
# 1-epoch, 10-batch smoke test of the OMol25 training pipeline on a single
# H200. Shakes out config/path/format issues before committing to the full
# multi-day training run.
#
# Prerequisite: preprocess_omol25.slurm has finished and produced the
# processed_data_dir referenced in configs/omol25_4m_cfm.yaml.

#SBATCH -J omol25_smoke
#SBATCH -p gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_smoke/smoke-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_smoke/smoke-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_smoke"
CONFIG="${1:-$PROJ/configs/omol25_4m_cfm.yaml}"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "host: $(hostname)"
echo "gpu:"; nvidia-smi -L
echo "config: $CONFIG"
echo ""

echo "=== processed dataset check ==="
python scripts/check_processed_dataset.py \
    --config "$CONFIG" \
    --split train \
    --batches 1
echo ""

# --fast = 1 epoch, 10 train batches, 1 val batch. Enough to catch:
#   - processed-data-dir path / file mismatches
#   - atom-map mismatch between config and processed tensors
#   - SampleAnalyzer failures
#   - OOM on H200 (should have huge headroom)
python scripts/run_train.py --config "$CONFIG" --fast
