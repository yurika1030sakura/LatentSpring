#!/bin/bash
#SBATCH -J bgfm-utility-onpolicy-audit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
PRODUCER="${4:?producer required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
CASES=(1 2 3 5)
INDEX="${CASES[$SLURM_ARRAY_TASK_ID]}"
printf -v PAD '%02d' "$INDEX"
"$STORE/envs/flowmol/bin/python" -u scripts/research/evaluate_utility_onpolicy.py \
  --phase audit --index "$INDEX" --project "$PROJECT" \
  --out "$RUN_ROOT/condition_$PAD" --run "$PRODUCER/condition_$PAD"
