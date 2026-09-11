#!/bin/bash
#SBATCH -J bgfm-eacf-audit
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
REPLICA="${3:?replica required}"
PROFILE="${4:-published}"
case "$PROFILE" in
  published) TRAIN_STEM=eacf_joint_condition_00 ;;
  compact) TRAIN_STEM=eacf_compact_condition_00 ;;
  *) exit 2 ;;
esac
LAB=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
EACF_SOURCE="$LAB/external/eacf_20260910"
[ "$(git -C "$EACF_SOURCE" rev-parse HEAD)" = beafab1b1ccd2b770572daeef1cf15f3fe199c21 ]
git -C "$EACF_SOURCE" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT:$EACF_SOURCE"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/eacf_audit_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/eacf_audit_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
exec "$LAB/external/eacf_env_20260910/bin/python" -u scripts/research/audit_eacf_joint_refinement.py \
  --run "$LAB/iclr2027/runs/${TRAIN_STEM}_s${REPLICA}_v1" \
  --source "$LAB/iclr2027/runs/eacf_source_condition_00_v1" --out "$RUN_ROOT/audit.json"
