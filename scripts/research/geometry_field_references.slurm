#!/bin/bash
#SBATCH -J ls-geometry-references
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?source}"; PROJECT="${2:?project}"; RUN="${3:?run}"
FLOW_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$FLOW_ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$SOURCE"
exec "$FLOW_ENV/bin/python" -s -u -m scripts.research.prepare_geometry_field_references \
 --project "$PROJECT" --protocol "$SOURCE/research/evidence/geometry_recovery_field_v1.json" --out "$RUN/references"
