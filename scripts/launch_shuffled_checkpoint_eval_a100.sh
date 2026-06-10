#!/bin/bash
# Evaluate one shuffled-force BGFM checkpoint.
#
# Environment:
#   STEP=50000|100000|150000|200000
# Usage:
#   STEP=50000 sbatch scripts/launch_shuffled_checkpoint_eval_a100.sh [n_samples]

#SBATCH -J shuffle_ckpt_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_shuffled_control/logs/eval-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_shuffled_control/logs/eval-%j.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
STEP="${STEP:-50000}"
N="${1:-${N:-200}}"
RUN_ROOT="$PROJ/runs/omol25_4m_bgfm_shuffled_from_fm"
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
OUT_ROOT="$PROJ/runs/eval/bgfm_shuffled_control"
OUT_DIR="$OUT_ROOT/shuffled_${STEP}"
mkdir -p "$OUT_ROOT/logs" "$OUT_DIR"
cd "$PROJ"

CKPT="$(find "$RUN_ROOT/lightning_logs" -type f -path "*/checkpoints/midstep-step=${STEP}.ckpt" | sort | tail -1 || true)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "checkpoint not found for STEP=$STEP under $RUN_ROOT" >&2
  exit 2
fi

echo "host: $(hostname)"
nvidia-smi -L
echo "step: $STEP"
echo "checkpoint: $CKPT"
echo "n_samples: $N"

OUT_DIR="$OUT_DIR" \
TABLE_CSV="$OUT_DIR/paper1_physics_table.csv" \
N_TIMESTEPS=100 \
BATCH_SIZE=16 \
FLOW_DEVICE=cuda \
OMOL25_DEVICE=cpu \
RELAX_STEPS=30 \
NO_DISCRETE=1 \
scripts/run_physics_eval_pipeline.sh \
  "cell_4d_bgfm_shuffled_${STEP}" \
  OMol25_4M \
  "BGFM_shuffled_${STEP}" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
