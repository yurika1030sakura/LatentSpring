#!/bin/bash
# Guidance-only ablation on the original FM checkpoint.
#
# Uses the same sampler guidance knobs as the corrected BGFM sweep, but without
# BGFM fine-tuning. If this fails to match BGFM, the paper can separate
# training-time force consistency from a generic sampling trick.
#
# Usage:
#   sbatch scripts/launch_fm_guidance_only_sweep_a100.sh [n_samples]

#SBATCH -J fm_guidance_only
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --array=0-2
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fm_guidance_only/logs/eval-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/fm_guidance_only/logs/eval-%A_%a.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CKPT="$PROJ/runs/omol25_4m_cfm/lightning_logs/version_8753404/checkpoints/epoch=0-step=292658.ckpt"
CONFIG="$PROJ/configs/omol25_4m_cfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
N="${1:-200}"
OUT_ROOT="$PROJ/runs/eval/fm_guidance_only"
mkdir -p "$OUT_ROOT/logs"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) WEIGHT="0.03"; START="0.90"; SCHEDULE="late_quadratic"; LABEL="fm_w003_s090_quad" ;;
  1) WEIGHT="0.03"; START="0.85"; SCHEDULE="late_linear";    LABEL="fm_w003_s085_lin" ;;
  2) WEIGHT="0.05"; START="0.90"; SCHEDULE="late_linear";    LABEL="fm_w005_s090_lin" ;;
  *) echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

OUT_DIR="$OUT_ROOT/$LABEL"
mkdir -p "$OUT_DIR"
cd "$PROJ"

echo "host: $(hostname)"
nvidia-smi -L
echo "checkpoint: $CKPT"
echo "n_samples: $N"
echo "guidance weight: $WEIGHT"
echo "guidance start: $START"
echo "guidance schedule: $SCHEDULE"

OUT_DIR="$OUT_DIR" \
TABLE_CSV="$OUT_DIR/paper1_physics_table.csv" \
N_TIMESTEPS=100 \
BATCH_SIZE=16 \
FLOW_DEVICE=cuda \
OMOL25_DEVICE=cpu \
RELAX_STEPS=30 \
NO_DISCRETE=1 \
BGFM_GUIDANCE_WEIGHT="$WEIGHT" \
BGFM_GUIDANCE_START="$START" \
BGFM_GUIDANCE_SCHEDULE="$SCHEDULE" \
BGFM_GUIDANCE_CLIP=5.0 \
BGFM_GUIDANCE_RATIO=1.0 \
scripts/run_physics_eval_pipeline.sh \
  "cell_4d_fm_guidance_only_${LABEL}" \
  OMol25_4M \
  "FM_guidance_only_${LABEL}" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
