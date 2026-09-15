#!/bin/bash
#SBATCH -J lspring-review-audit
#SBATCH -A ryl_lab
#SBATCH -p serial_requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?immutable source}"
PROJECT="${2:?project data root}"
RUN_ROOT="${3:?run root}"
KIND="${4:?factorial, replication or matched_physical}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
cd "$SOURCE"
if [[ "$KIND" == factorial ]]; then
 script=scripts/research/audit_source_physical_factorial.py
elif [[ "$KIND" == replication ]]; then
 script=scripts/research/audit_source_replication.py
elif [[ "$KIND" == matched_physical ]]; then
 script=scripts/research/audit_matched_physical.py
else
 exit 2
fi
exec "$FLOWMOL_ENV/bin/python" -u "$script" --project "$PROJECT" --run "$RUN_ROOT" --out "$RUN_ROOT/audit.json"
