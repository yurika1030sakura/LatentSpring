#!/bin/bash
# Targeted sampling-guidance sweep for the 400k OMol25 BGFM checkpoint.
#
# The 250k result improved raw force but worsened relaxation drop. This sweep
# tests weaker and later BGFM score guidance settings to preserve the force
# gain while reducing terminal strain.
#
# Usage:
#   sbatch scripts/launch_400k_guidance_sweep_a100.sh [n_samples]

#SBATCH -J bgfm400_guidance
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --array=0-7
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm400_guidance/logs/eval-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm400_guidance/logs/eval-%A_%a.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CKPT="$PROJ/runs/omol25_4m_bgfm/lightning_logs/version_10721924/checkpoints/midstep-step=400000.ckpt"
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
N="${1:-100}"
OUT_ROOT="$PROJ/runs/eval/bgfm400_guidance"
mkdir -p "$OUT_ROOT/logs"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) WEIGHT="0.03"; START="0.75"; SCHEDULE="late_linear";    LABEL="w003_s075_lin" ;;
  1) WEIGHT="0.05"; START="0.75"; SCHEDULE="late_linear";    LABEL="w005_s075_lin" ;;
  2) WEIGHT="0.08"; START="0.75"; SCHEDULE="late_linear";    LABEL="w008_s075_lin" ;;
  3) WEIGHT="0.10"; START="0.75"; SCHEDULE="late_linear";    LABEL="w010_s075_lin" ;;
  4) WEIGHT="0.05"; START="0.80"; SCHEDULE="late_linear";    LABEL="w005_s080_lin" ;;
  5) WEIGHT="0.08"; START="0.80"; SCHEDULE="late_linear";    LABEL="w008_s080_lin" ;;
  6) WEIGHT="0.05"; START="0.70"; SCHEDULE="late_quadratic"; LABEL="w005_s070_quad" ;;
  7) WEIGHT="0.08"; START="0.70"; SCHEDULE="late_quadratic"; LABEL="w008_s070_quad" ;;
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
  "cell_4d_400k_${LABEL}" \
  OMol25_4M \
  "BGFM_400k_${LABEL}" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
