#!/bin/bash
#SBATCH -J bgfm-shape-reference
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
BGFM_REFERENCE_POWER="${3:-11}"
BGFM_REFERENCE_SCRAMBLES="${4:-4}"
BGFM_REUSE_REFERENCE="${5:-}"
BGFM_STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$BGFM_STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_shape_ref_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_shape_ref_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
BGFM_REUSE_ARGS=()
if [[ -n "$BGFM_REUSE_REFERENCE" ]]; then BGFM_REUSE_ARGS+=(--reuse-reference "$BGFM_REUSE_REFERENCE"); fi
exec "$BGFM_STORE/envs/flowmol/bin/python" -u scripts/research/refined_triatomic_reference.py \
 --pilot "$LAB_RUNS/triatomic_reference_1137_v1" --metadata "$LAB_RUNS/raw_metadata_replay_v1" \
 --oracle "$BGFM_STORE/checkpoints/omol25/esen_sm_conserving_all.pt" \
 --oracle-python "$BGFM_STORE/envs/omol25/bin/python" --out "$RUN_ROOT" \
 --power "$BGFM_REFERENCE_POWER" --scrambles "$BGFM_REFERENCE_SCRAMBLES" "${BGFM_REUSE_ARGS[@]}"
