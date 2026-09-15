#!/bin/bash
#SBATCH -J lspring-fixed-schedule
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE="${1:?immutable source}"
PROJECT="${2:?project data root}"
RUN_ROOT="${3:?run root}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONPATH="$SOURCE" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/lspring_fixed_schedule_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/lspring_fixed_schedule_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE"
for seed in 0 1; do
 protocol="$SOURCE/research/evidence/matched_physical_s${seed}_v2.json"
 "$FLOWMOL_ENV/bin/python" -u scripts/research/repair_fixed_gaga_physical.py \
  --project "$PROJECT" --protocol "$protocol" --out "$RUN_ROOT/s${seed}/study"
 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 "$FLOWMOL_ENV/bin/python" -u scripts/research/matched_physical_xtb.py \
  --project "$PROJECT" --protocol "$protocol" --generation "$RUN_ROOT/s${seed}/study" \
  --out "$RUN_ROOT/s${seed}/xtb" --workers 8
done
