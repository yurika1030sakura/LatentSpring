#!/bin/bash
#SBATCH -J bgfm-jax-oracle
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
LAB=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
EACF_SOURCE="$LAB/external/eacf_20260910"
EACF_PY="$LAB/external/eacf_env_20260910/bin/python"
[ "$(git -C "$EACF_SOURCE" rev-parse HEAD)" = beafab1b1ccd2b770572daeef1cf15f3fe199c21 ]
git -C "$EACF_SOURCE" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT:$EACF_SOURCE"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
export MPLBACKEND=Agg WANDB_MODE=disabled
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/jax_oracle_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/jax_oracle_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$EACF_PY" -u scripts/research/qualify_jax_oracle_bridge.py \
 --panel "$LAB/iclr2027/runs/jax_oracle_panel_v1" --out "$RUN_ROOT" \
 --oracle-python "$STORE/envs/omol25/bin/python" \
 --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
