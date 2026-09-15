#!/bin/bash
#SBATCH -J lspring-source-repeat
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?immutable original experiment source}"
PROJECT="${2:?project data root}"
RUN_ROOT="${3:?run root}"
for seed in 2 3 4; do
 export SLURM_ARRAY_TASK_ID="$seed"
 bash "$SOURCE/scripts/research/source_replication.slurm" "$SOURCE" "$PROJECT" "$RUN_ROOT"
done
