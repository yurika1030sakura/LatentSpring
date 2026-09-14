#!/bin/bash
#SBATCH -J bgfm-tree-audit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
PROJECT="${3:?project required}"
PRIOR="${4:?prior run required}"
FIXED="${5:?fixed run required}"
LEARNED="${6:?learned run required}"
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/audit_tree_prior_study.py \
 --protocol research/evidence/tree_prior_protocol_v2.json --prior "$PRIOR/prior" \
 --fixed "$FIXED/study" --learned "$LEARNED/study" --out "$RUN_ROOT/audit.json"
