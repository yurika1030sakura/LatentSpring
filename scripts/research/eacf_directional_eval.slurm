#!/bin/bash
#SBATCH -J bgfm-eacf-direction-eval
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
PROJECT="${3:?artifact root required}"
LAB=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
UPSTREAM="$LAB/external/eacf_20260910"
[ "$(git -C "$UPSTREAM" rev-parse HEAD)" = beafab1b1ccd2b770572daeef1cf15f3fe199c21 ]
git -C "$UPSTREAM" diff --quiet HEAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/cache" CUDA_CACHE_PATH="$RUN_ROOT/runtime/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$SOURCE_ROOT"
for PROFILE in published compact; do
  for REPLICA in 0 1; do
    "$STORE/envs/flowmol/bin/python" -u scripts/research/evaluate_eacf_directional.py \
      --flows "$PROJECT/runs" --table "$PROJECT/runs/chemical_policy_table_v1" \
      --controls "$PROJECT/runs/normalized_site_eval_v1" --interfaces "$PROJECT/runs/eacf_directional_interface_qualified_v1" \
      --out "$RUN_ROOT/${PROFILE}_s${REPLICA}" --profile "$PROFILE" --replica "$REPLICA" \
      --transport-python "$LAB/external/eacf_env_20260910/bin/python" --upstream "$UPSTREAM" \
      --oracle-python "$STORE/envs/omol25/bin/python" --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
  done
done
