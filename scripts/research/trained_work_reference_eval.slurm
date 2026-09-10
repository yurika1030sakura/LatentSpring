#!/bin/bash
#SBATCH -J bgfm-frozen-work-eval
#SBATCH -A ryl_lab
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?output required}"
TRAINING="${3:?completed trained proposal required}"
BGFM_EVAL_PARTICLES="${4:-4096}"
BGFM_EVAL_BATCH="${5:-64}"
BGFM_SOURCE_STREAM="${6:-0}"
BGFM_REFERENCE="${7:-/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs/refined_triatomic_reference_1137_v1}"
BGFM_EVAL_SEED="${8:-9065}"
FLOWMOL_ENV=/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export LD_LIBRARY_PATH="$FLOWMOL_ENV/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_frozen_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_frozen_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
BGFM_STREAM_ARGS=()
if [[ "$BGFM_SOURCE_STREAM" == 1 ]]; then BGFM_STREAM_ARGS+=(--source-evaluation-stream); fi
"$FLOWMOL_ENV/bin/python" -u scripts/research/sample_trained_work.py --run "$TRAINING" --out "$RUN_ROOT/sampling" \
 --particles "$BGFM_EVAL_PARTICLES" --batch "$BGFM_EVAL_BATCH" --seed "$BGFM_EVAL_SEED" "${BGFM_STREAM_ARGS[@]}"
exec "$FLOWMOL_ENV/bin/python" -u scripts/research/compare_work_reference.py --run "$RUN_ROOT/sampling" \
 --reference "$BGFM_REFERENCE" \
 --out "$RUN_ROOT/normalizer.json" --stages final
