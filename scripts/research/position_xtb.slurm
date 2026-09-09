#!/bin/bash
#SBATCH -J bgfm-position-xtb
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$SOURCE_ROOT"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PATH="$FLOWMOL_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$SOURCE_ROOT"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/eval_position_xtb.py \
 --samples /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/position_samples_v1/samples.json \
 --out "$RUN_ROOT" --workers 4 --max-cycles 200
