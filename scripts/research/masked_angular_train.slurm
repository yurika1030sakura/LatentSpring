#!/bin/bash
#SBATCH -J bgfm-masked-angle-train
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
TABLE=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/chemical_policy_table_v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
MODELS=(tensor vector)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID%2))
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/train_masked_angular.py \
  --table "$TABLE" --out "$RUN_ROOT/${MODEL}_s${REPLICA}" --model "$MODEL" --replica "$REPLICA"
