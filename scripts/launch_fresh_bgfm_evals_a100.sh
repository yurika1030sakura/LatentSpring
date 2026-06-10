#!/bin/bash
# Fresh-checkpoint physics evals for the active BGFM runs.
#
# Array tasks:
#   0: OMol25 BGFM 250k, unguided
#   1: OMol25 BGFM 250k, guided w=0.20
#   2: Cell B GEOM-BGFM epoch 5, unguided
#
# Usage:
#   sbatch scripts/launch_fresh_bgfm_evals_a100.sh [n_samples]

#SBATCH -J fresh_bgfm_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fresh_bgfm/logs/eval-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fresh_bgfm/logs/eval-%A_%a.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
N="${1:-200}"
mkdir -p "$PROJ/runs/eval/fresh_bgfm/logs"
cd "$PROJ"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    CELL_ID="cell_4d_250k"
    DATA_LABEL="OMol25_4M"
    METHOD_LABEL="BGFM_250k"
    CKPT="$PROJ/runs/omol25_4m_bgfm/lightning_logs/version_10721924/checkpoints/midstep-step=250000.ckpt"
    CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
    EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
    OUT_DIR="$PROJ/runs/eval/paper1_bgfm_250k"
    GUIDANCE_WEIGHT="0.0"
    ;;
  1)
    CELL_ID="cell_4d_250k_guided_w020"
    DATA_LABEL="OMol25_4M"
    METHOD_LABEL="BGFM_250k_guided_w020"
    CKPT="$PROJ/runs/omol25_4m_bgfm/lightning_logs/version_10721924/checkpoints/midstep-step=250000.ckpt"
    CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
    EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
    OUT_DIR="$PROJ/runs/eval/paper1_bgfm_250k_guided_w020"
    GUIDANCE_WEIGHT="0.20"
    ;;
  2)
    CELL_ID="cell_B_e5"
    DATA_LABEL="GEOM_OMol25"
    METHOD_LABEL="BGFM_e5"
    CKPT="$PROJ/runs/geom_bgfm/lightning_logs/version_10721923/checkpoints/epoch=5-step=131688.ckpt"
    CONFIG="$PROJ/configs/geom_bgfm.yaml"
    EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/geom_bgfm_processed
    OUT_DIR="$PROJ/runs/eval/paper1_cellB_e5"
    GUIDANCE_WEIGHT="0.0"
    ;;
  *)
    echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
echo "host: $(hostname)"
nvidia-smi -L
echo "cell: $CELL_ID"
echo "checkpoint: $CKPT"
echo "n_samples: $N"
echo "guidance_weight: $GUIDANCE_WEIGHT"

OUT_DIR="$OUT_DIR" \
TABLE_CSV="$OUT_DIR/paper1_physics_table.csv" \
N_TIMESTEPS=100 \
BATCH_SIZE=16 \
FLOW_DEVICE=cuda \
OMOL25_DEVICE=cpu \
RELAX_STEPS=30 \
NO_DISCRETE=1 \
BGFM_GUIDANCE_WEIGHT="$GUIDANCE_WEIGHT" \
BGFM_GUIDANCE_START=0.75 \
BGFM_GUIDANCE_SCHEDULE=late_linear \
BGFM_GUIDANCE_CLIP=5.0 \
BGFM_GUIDANCE_RATIO=1.0 \
scripts/run_physics_eval_pipeline.sh \
  "$CELL_ID" \
  "$DATA_LABEL" \
  "$METHOD_LABEL" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
