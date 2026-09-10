#!/bin/bash
#SBATCH -J bgfm-box-reference
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
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_box_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_box_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
BGFM_ORDER_ARGS=()
if [[ $# -gt 2 ]]; then BGFM_ORDER_ARGS=(--orders "${@:3}"); fi
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/triatomic_box_reference.py \
 --reference /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/refined_triatomic_reference_1137_v2 --out "$RUN_ROOT" "${BGFM_ORDER_ARGS[@]}"
