#!/bin/bash
#SBATCH -J bgfm-fragment-physical
#SBATCH -A ryl_lab
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?source required}"
RUN_ROOT="${2:?unique output required}"
INPUTS="${3:?frozen source panel required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0
export TMPDIR="$RUN_ROOT/runtime/tmp" MPLCONFIGDIR="$RUN_ROOT/runtime/mpl"
export XDG_CACHE_HOME="$RUN_ROOT/runtime/cache" CUDA_CACHE_PATH="$RUN_ROOT/runtime/cuda"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$SOURCE_ROOT"
for INDEX in 1 2 4 6; do
  WARM="$RUN_ROOT/warm_${INDEX}"
  "$STORE/envs/flowmol/bin/python" -u scripts/research/fragment_physical_pilot.py \
    --stage warm --condition-index "$INDEX" --source "$INPUTS" --out "$WARM" \
    --oracle-python "$STORE/envs/omol25/bin/python" --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
  for METHOD in local_only all_singletons fragments4; do
    for REPLICA in 0 1; do
      "$STORE/envs/flowmol/bin/python" -u scripts/research/fragment_physical_pilot.py \
        --stage evaluate --condition-index "$INDEX" --source "$INPUTS" --warm "$WARM" \
        --method "$METHOD" --replica "$REPLICA" --out "$RUN_ROOT/condition_${INDEX}/${METHOD}_s${REPLICA}" \
        --oracle-python "$STORE/envs/omol25/bin/python" --oracle-checkpoint "$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
    done
  done
done
