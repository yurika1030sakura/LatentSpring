#!/bin/bash
#SBATCH -J bgfm-site-concentration-audit
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
CONTROL_RUN="${4:?control directory required}"
CONCENTRATIONS=(64 400)
KAPPA="${CONCENTRATIONS[$((SLURM_ARRAY_TASK_ID/2))]}"
REPLICA=$((SLURM_ARRAY_TASK_ID%2))
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
for CASE in primary transfer_00 transfer_01 transfer_02 transfer_03 transfer_04 transfer_05; do
  /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/audit_fresh_reuse.py \
    --protocol "research/evidence/site_concentration_${KAPPA}_${CASE}_v1.json" --project "$PROJECT" \
    --run "$CONTROL_RUN/kappa_$KAPPA/$CASE/replica_$REPLICA" --out "$RUN_ROOT/kappa_$KAPPA/$CASE/replica_$REPLICA/results.json"
done
