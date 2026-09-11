#!/bin/bash
#SBATCH -J bgfm-eacf-smoke
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?committed source snapshot required}"
RUN_ROOT="${2:?unique output required}"
EACF_SOURCE=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_20260910
EACF_PY=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_env_20260910/bin/python
EXPECTED_REV=beafab1b1ccd2b770572daeef1cf15f3fe199c21
[ "$(git -C "$EACF_SOURCE" rev-parse HEAD)" = "$EXPECTED_REV" ]
git -C "$EACF_SOURCE" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$EACF_SOURCE"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
export MPLBACKEND=Agg WANDB_MODE=disabled
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/eacf_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/eacf_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
"$EACF_PY" -c 'import jax,json; print(json.dumps({"jax":jax.__version__,"devices":[str(x) for x in jax.devices()],"scope":"upstream engineering only; no molecular baseline"}))'
exec "$EACF_PY" -m pytest \
  "$EACF_SOURCE/eacf/flow/aug_flow_dist_test.py::test_distribution" \
  "$EACF_SOURCE/eacf/flow/bijectors/spherical_coupling_test.py::test_bijector_spherical" \
  -q -o "cache_dir=$RUN_ROOT/pytest_cache" --junitxml="$RUN_ROOT/junit.xml"
