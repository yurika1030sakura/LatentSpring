#!/bin/bash
#SBATCH -J bgfm-split-audit
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$SOURCE_ROOT"
cd "$SOURCE_ROOT"
exec /n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u \
 scripts/research/audit_composition_split.py \
 --root /n/holylabs/woo_lab/Lab/yulili/bgfm/processed_data/omol25_4m_processed \
 --out "$RUN_ROOT/composition_split.json"
