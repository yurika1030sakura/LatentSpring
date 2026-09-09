#!/bin/bash
#SBATCH -J bgfm-work-xtb
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
BASELINE="${3:?baseline samples required}"
CANDIDATE="${4:?candidate samples required}"
CONTROL="${5:-}"
BASELINE_LABEL="${6:-initial}"
CANDIDATE_LABEL="${7:-joint}"
CONTROL_LABEL="${8:-energy}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$SOURCE_ROOT"
export PATH="$FLOWMOL_ENV/bin:$PATH" LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_work_eval_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_work_eval_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
BGFM_EXTRA_ARGS=()
if [[ -n "$CONTROL" ]]; then BGFM_EXTRA_ARGS+=(--samples "$CONTROL_LABEL" "$CONTROL"); fi
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/assess_work_panel.py \
 --samples "$BASELINE_LABEL" "$BASELINE" --samples "$CANDIDATE_LABEL" "$CANDIDATE" "${BGFM_EXTRA_ARGS[@]}" \
 --out "$RUN_ROOT" --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
