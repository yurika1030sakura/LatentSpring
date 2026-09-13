#!/bin/bash
#SBATCH -J bgfm-work-single-controls-audit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%A_%a.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?project required}"
INPUT="${4:?evaluation run required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
INDICES=(0 1 2 3 5 7)
INDEX="${INDICES[$SLURM_ARRAY_TASK_ID]}"
printf -v CONDITION 'condition_%02d' "$INDEX"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/evaluate_work_chains.py \
 --project "$PROJECT" --out "$RUN_ROOT/$CONDITION" --index "$INDEX" --protocol research/evidence/work_chain_single_controls_protocol_v1.json --phase audit --run "$INPUT/$CONDITION"
