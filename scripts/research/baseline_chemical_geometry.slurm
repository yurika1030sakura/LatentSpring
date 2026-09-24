#!/bin/bash
#SBATCH -J ls-baseline-geometry
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?source}"; PROJECT="${2:?project}"; OUT="${3:?out}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$SOURCE"
exec "$FLOW_ENV/bin/python" -s -u -m scripts.research.audit_baseline_chemical_geometry --project "$PROJECT" --out "$OUT"
