#!/bin/bash
#SBATCH -J bgfm-raw-metadata
#SBATCH -A ryl_lab
#SBATCH -p serial_requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONPATH="$SOURCE_ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$RUN_ROOT"
cd "$SOURCE_ROOT"
REPLAY_ARGS=()
if [ -f "$RUN_ROOT/source_index.sqlite" ]; then REPLAY_ARGS+=(--resume); fi
exec "$BGFM_STORE/envs/omol25/bin/python" -u scripts/research/recover_raw_metadata.py \
 --raw "$BGFM_STORE/raw_data/omol25/v250514/train_4M_extracted/train_4M" \
 --processed "$BGFM_STORE/processed_data/omol25_4m_processed" \
 --archive-manifest "$BGFM_STORE/raw_data/omol25/v250514/download_manifest.json" --out "$RUN_ROOT" "${REPLAY_ARGS[@]}"
