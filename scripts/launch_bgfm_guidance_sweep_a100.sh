#!/bin/bash
# Sweep sampling-time BGFM score guidance on the fixed OMol25 BGFM checkpoint.
#
# This does not retrain the model. It tests whether the force-aligned score
# learned by BGFM can directly improve terminal sample physics when used as a
# late-time normalized guidance drift during Euler sampling.
#
# Usage:
#   sbatch scripts/launch_bgfm_guidance_sweep_a100.sh [checkpoint] [n_samples]

#SBATCH -J bgfm_guided_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/paper1_bgfm_guidance_sweep/logs/guided-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/paper1_bgfm_guidance_sweep/logs/guided-%A_%a.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CKPT="${1:-$PROJ/runs/omol25_4m_bgfm/lightning_logs/version_9701914/checkpoints/midstep-step=200000.ckpt}"
N="${2:-100}"
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
OUT_DIR="$PROJ/runs/eval/paper1_bgfm_guidance_sweep"
mkdir -p "$OUT_DIR/logs"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) WEIGHT="0.05"; LABEL="w005" ;;
  1) WEIGHT="0.10"; LABEL="w010" ;;
  2) WEIGHT="0.20"; LABEL="w020" ;;
  *) echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

cd "$PROJ"
echo "host: $(hostname)"
nvidia-smi -L
echo "checkpoint: $CKPT"
echo "guidance weight: $WEIGHT"
echo "n_samples: $N"

OUT_DIR="$OUT_DIR" \
TABLE_CSV="$OUT_DIR/paper1_physics_table_${LABEL}.csv" \
N_TIMESTEPS=100 \
BATCH_SIZE=16 \
FLOW_DEVICE=cuda \
OMOL25_DEVICE=cpu \
RELAX_STEPS=30 \
NO_DISCRETE=1 \
BGFM_GUIDANCE_WEIGHT="$WEIGHT" \
BGFM_GUIDANCE_START=0.75 \
BGFM_GUIDANCE_SCHEDULE=late_linear \
BGFM_GUIDANCE_CLIP=5.0 \
BGFM_GUIDANCE_RATIO=1.0 \
scripts/run_physics_eval_pipeline.sh \
  "cell_4d_guided_${LABEL}" \
  OMol25_4M \
  "BGFM_guided_${LABEL}" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
