#!/bin/bash
#SBATCH -J bgfm-reference-pilot
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
MODE="${3:-smoke}"
DOMAIN="${4:-unrestricted}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
BGFM_RUNTIME="$RUN_ROOT/runtime"
export TMPDIR="$BGFM_RUNTIME/tmp" MPLCONFIGDIR="$BGFM_RUNTIME/mpl"
export XDG_CACHE_HOME="$BGFM_RUNTIME/cache" CUDA_CACHE_PATH="$BGFM_RUNTIME/cuda"
export TRITON_CACHE_DIR="$BGFM_RUNTIME/triton"
mkdir -p "$RUN_ROOT" "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
cd "$SOURCE_ROOT"
ARGS=(--out "$RUN_ROOT" --runs-root "$LAB_RUNS" --domain "$DOMAIN" --oracle-python "$STORE/envs/omol25/bin/python"
      --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt")
if [ "$MODE" = smoke ]; then ARGS+=(--engineering-smoke); elif [ "$MODE" != pilot ]; then exit 2; fi
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/parity_reference_pilot.py "${ARGS[@]}"
