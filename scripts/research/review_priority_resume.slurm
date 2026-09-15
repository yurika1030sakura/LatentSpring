#!/bin/bash
#SBATCH -J lspring-review-priority
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
PHYSICAL_SOURCE="${1:?immutable physical source}"
GAGA_SOURCE="${2:?immutable GAGA source}"
PROJECT="${3:?project data root}"
PHYSICAL_RUN="${4:?physical run root}"
GAGA_RUN="${5:?GAGA training run root}"
bash "$PHYSICAL_SOURCE/scripts/research/source_physical_factorial.slurm" \
 "$PHYSICAL_SOURCE" "$PROJECT" "$PHYSICAL_RUN"
export SLURM_ARRAY_TASK_ID=0
exec bash "$GAGA_SOURCE/scripts/research/gaga_feedback.slurm" \
 "$GAGA_SOURCE" "$PROJECT" "$GAGA_RUN"
