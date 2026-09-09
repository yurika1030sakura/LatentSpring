#!/bin/bash
#SBATCH -J bgfm-estimator-smokes
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
for arm in squared_independent product_independent squared_common product_common; do
  bash "$SOURCE_ROOT/scripts/research/runtime_smoke.slurm" "$SOURCE_ROOT" \
    "$RUN_ROOT/$arm" "configs/research/smoke_${arm}.yaml" 9001
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python \
    "$SOURCE_ROOT/scripts/research/summarize_training.py" "$RUN_ROOT/$arm"
done
