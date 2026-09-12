#!/bin/bash
#SBATCH -J bgfm-transfer-reuse-audit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
EVALUATION_RUN="${4:?evaluation directory required}"
METHODS=(site learned_vector)
METHOD="${METHODS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID%2))
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
for INDEX in 00 01 02 03 04 05; do
  "$STORE/envs/flowmol/bin/python" -u scripts/research/audit_fresh_reuse.py \
    --protocol "research/evidence/transfer_reuse_case_$INDEX.json" --project "$PROJECT" \
    --run "$EVALUATION_RUN/condition_$INDEX/${METHOD}_s${REPLICA}" \
    --out "$RUN_ROOT/condition_$INDEX/${METHOD}_s${REPLICA}/results.json"
done
