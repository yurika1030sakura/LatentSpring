#!/bin/bash
#SBATCH -J bgfm-parity-mcmc
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
KERNEL="${3:?mala or hmc required}"
REPLICA="${4:-0}"
MODE="${5:-production}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/parity_mcmc_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/parity_mcmc_cache_${SLURM_JOB_ID}"
mkdir -p "$RUN_ROOT" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
cd "$SOURCE_ROOT"
ARGS=(--samples "$LAB_RUNS/parity_cpu_gpu_control_s0_v1/base_samples.pt" --out "$RUN_ROOT"
      --kernel "$KERNEL" --replica "$REPLICA"
      --oracle-python "$STORE/envs/omol25/bin/python"
      --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt")
if [ "$MODE" = smoke ]; then ARGS+=(--engineering-smoke);
elif [ "$MODE" != production ]; then exit 2; fi
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/parity_mcmc_control.py "${ARGS[@]}"
