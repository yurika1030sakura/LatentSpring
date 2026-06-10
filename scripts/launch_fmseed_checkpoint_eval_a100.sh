#!/bin/bash
# Evaluate one checkpoint from the clean FM-seeded comparison.
#
# Environment:
#   MODE=bgfm|fm
#   STEP=50000|100000|150000|200000
# Usage:
#   MODE=bgfm STEP=50000 sbatch scripts/launch_fmseed_checkpoint_eval_a100.sh [n_samples]

#SBATCH -J fmseed_ckpt_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fmseed_vs_control/logs/eval-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fmseed_vs_control/logs/eval-%j.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
MODE="${MODE:-bgfm}"
STEP="${STEP:-50000}"
N="${1:-${N:-200}}"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
OUT_ROOT="$PROJ/runs/eval/fmseed_vs_control"
mkdir -p "$OUT_ROOT/logs"
cd "$PROJ"

case "$MODE" in
  bgfm)
    RUN_ROOT="$PROJ/runs/omol25_4m_bgfm_dirft_from_fm"
    CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
    CELL_ID="cell_4d_bgfm_fmseed_${STEP}"
    METHOD_LABEL="BGFM_fmseed_${STEP}"
    OUT_DIR="$OUT_ROOT/bgfm_${STEP}"
    ;;
  fm)
    RUN_ROOT="$PROJ/runs/omol25_4m_cfm_continue_from_fm"
    CONFIG="$PROJ/configs/omol25_4m_cfm.yaml"
    CELL_ID="cell_4d_fm_continue_${STEP}"
    METHOD_LABEL="FM_continue_${STEP}"
    OUT_DIR="$OUT_ROOT/fm_continue_${STEP}"
    ;;
  *)
    echo "MODE must be bgfm or fm, got: $MODE" >&2
    exit 2
    ;;
esac

CKPT="$(find "$RUN_ROOT/lightning_logs" -type f -path "*/checkpoints/midstep-step=${STEP}.ckpt" | sort | tail -1 || true)"
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "checkpoint not found for MODE=$MODE STEP=$STEP under $RUN_ROOT" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "host: $(hostname)"
nvidia-smi -L
echo "mode: $MODE"
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
  "$CELL_ID" \
  OMol25_4M \
  "$METHOD_LABEL" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
