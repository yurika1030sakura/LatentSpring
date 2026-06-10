#!/bin/bash
# Confirmation eval for the best 100-sample guided BGFM setting.
#
# Uses the fixed OMol25 BGFM 200k checkpoint with sampling-time BGFM score
# guidance weight 0.20. This run is meant to be compared directly against the
# existing 200-sample FM and unguided BGFM rows.
#
# Usage:
#   sbatch scripts/launch_bgfm_guidance_w020_final_a100.sh [checkpoint] [n_samples]

#SBATCH -J bgfm_w020_final
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/paper1_bgfm_guidance_w020_final/logs/final-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/paper1_bgfm_guidance_w020_final/logs/final-%j.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CKPT="${1:-$PROJ/runs/omol25_4m_bgfm/lightning_logs/version_9701914/checkpoints/midstep-step=200000.ckpt}"
N="${2:-200}"
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
OUT_DIR="$PROJ/runs/eval/paper1_bgfm_guidance_w020_final"
mkdir -p "$OUT_DIR/logs"

cd "$PROJ"
echo "host: $(hostname)"
nvidia-smi -L
echo "checkpoint: $CKPT"
echo "guidance weight: 0.20"
echo "n_samples: $N"

OUT_DIR="$OUT_DIR" \
TABLE_CSV="$OUT_DIR/paper1_physics_table.csv" \
N_TIMESTEPS=100 \
BATCH_SIZE=16 \
FLOW_DEVICE=cuda \
OMOL25_DEVICE=cpu \
RELAX_STEPS=30 \
NO_DISCRETE=1 \
BGFM_GUIDANCE_WEIGHT=0.20 \
BGFM_GUIDANCE_START=0.75 \
BGFM_GUIDANCE_SCHEDULE=late_linear \
BGFM_GUIDANCE_CLIP=5.0 \
BGFM_GUIDANCE_RATIO=1.0 \
scripts/run_physics_eval_pipeline.sh \
  cell_4d_guided_w020_final \
  OMol25_4M \
  BGFM_guided_w020 \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
