#!/bin/bash
#SBATCH -J bgfm-normalized-site-train
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-3%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
TABLE="${3:?training table required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
MODELS=(mixture vector)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID%2))
cd "$SOURCE_ROOT"
"$STORE/envs/flowmol/bin/python" -u scripts/research/train_normalized_site.py \
  --table "$TABLE" --out "$RUN_ROOT/${MODEL}_s${REPLICA}" --model "$MODEL" --replica "$REPLICA"
