#!/bin/bash
#SBATCH -J bgfm-eacf-joint
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
REPLICA="${4:-0}"
LAB=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
EACF_SOURCE="$LAB/external/eacf_20260910"
EACF_PY="$LAB/external/eacf_env_20260910/bin/python"
[ "$(git -C "$EACF_SOURCE" rev-parse HEAD)" = beafab1b1ccd2b770572daeef1cf15f3fe199c21 ]
git -C "$EACF_SOURCE" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT:$EACF_SOURCE"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
export NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg WANDB_MODE=disabled
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/eacf_joint_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/eacf_joint_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
ARGS=(--source "$LAB/iclr2027/runs/eacf_source_condition_00_v1" --out "$RUN_ROOT" --replica "$REPLICA"
      --oracle-python "$STORE/envs/omol25/bin/python" --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt")
if [ "$MODE" = smoke ]; then ARGS+=(--engineering-smoke); elif [ "$MODE" != production ]; then exit 2; fi
exec "$EACF_PY" -u scripts/research/train_eacf_joint_refinement.py "${ARGS[@]}"
