#!/bin/bash
#SBATCH -J bgfm-tree-energy
#SBATCH -A ryl_lab
#SBATCH -p sapphire,shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
PROJECT="${3:?project required}"
AUDIT="${4:?audit file required}"
FIXED="${5:?fixed run required}"
LEARNED="${6:?learned run required}"
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$SOURCE_ROOT"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_tree_energy_cache_${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_tree_energy_mpl_${SLURM_JOB_ID}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u scripts/research/tree_prior_energy.py \
 --protocol research/evidence/tree_prior_energy_protocol_v1.json --audit "$AUDIT" \
 --fixed "$FIXED/study" --learned "$LEARNED/study" --out "$RUN_ROOT/energy"
