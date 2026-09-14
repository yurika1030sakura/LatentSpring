#!/bin/bash
#SBATCH -J bgfm-tree-prior-fit
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
PROJECT="${3:?project required}"
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/train_tree_mixture_prior.py \
 --project "$PROJECT" --out "$RUN_ROOT/prior" --protocol research/evidence/tree_prior_protocol_v1.json
