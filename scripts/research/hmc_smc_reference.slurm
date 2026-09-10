#!/bin/bash
#SBATCH -J bgfm-hmc-smc
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
CONDITION_RUN="${3:?completed condition source required}"
BGFM_REFERENCE="${4:-}"
BGFM_REFERENCE_SEED="${5:-9069}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_hmc_smc_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_hmc_smc_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
BGFM_REFERENCE_ARGS=()
if [[ -n "$BGFM_REFERENCE" ]]; then BGFM_REFERENCE_ARGS+=(--reference "$BGFM_REFERENCE"); fi
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/hmc_smc_reference.py \
 --condition-run "$CONDITION_RUN" --out "$RUN_ROOT" --seed "$BGFM_REFERENCE_SEED" "${BGFM_REFERENCE_ARGS[@]}"
