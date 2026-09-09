#!/bin/bash
#SBATCH -J bgfm-official-val-audit
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$RUN_ROOT"
cd "$SOURCE_ROOT"
exec "$BGFM_STORE/envs/omol25/bin/python" -u scripts/research/audit_official_validation.py \
 --raw-val "$BGFM_STORE/raw_data/omol25/v250514/official_validation/val_extracted/val" \
 --processed "$BGFM_STORE/processed_data/omol25_4m_processed" \
 --training-index /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/raw_metadata_replay_v1/source_index_readonly.sqlite \
 --validation-manifest research/evidence/official_validation_recovery.json --out "$RUN_ROOT"
