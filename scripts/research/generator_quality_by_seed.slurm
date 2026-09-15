#!/bin/bash
#SBATCH -J lspring-quality
#SBATCH --array=0-1%2
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source}"
seed="$SLURM_ARRAY_TASK_ID"
for chunk in 0 1 2 3; do
 SLURM_ARRAY_TASK_ID=$((seed*4+chunk)) bash "$SOURCE_ROOT/scripts/research/generator_quality.slurm" "$@"
done
