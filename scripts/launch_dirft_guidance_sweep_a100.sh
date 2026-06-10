#!/bin/bash
# Targeted sampling-guidance sweep for corrected BGFM direct fine-tuning checkpoints.
#
# This is intentionally narrow: the un-guided 150k/200k corrected checkpoints are
# already the strongest force/relaxation operating points. We only test weak,
# late guidance settings that might improve force tails without reintroducing
# terminal strain.
#
# Usage:
#   sbatch scripts/launch_dirft_guidance_sweep_a100.sh [n_samples]

#SBATCH -J bgfm_dirft_guidance
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=192G
#SBATCH -t 08:00:00
#SBATCH --array=0-5
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_dirft_guidance/logs/eval-%A_%a.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/bgfm_dirft_guidance/logs/eval-%A_%a.err

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
EVAL_DATA=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
N="${1:-100}"
OUT_ROOT="$PROJ/runs/eval/bgfm_dirft_guidance"
mkdir -p "$OUT_ROOT/logs"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) STEP="150000"; WEIGHT="0.03"; START="0.85"; SCHEDULE="late_linear";    LABEL="150k_w003_s085_lin" ;;
  1) STEP="150000"; WEIGHT="0.05"; START="0.90"; SCHEDULE="late_linear";    LABEL="150k_w005_s090_lin" ;;
  2) STEP="150000"; WEIGHT="0.03"; START="0.90"; SCHEDULE="late_quadratic"; LABEL="150k_w003_s090_quad" ;;
  3) STEP="200000"; WEIGHT="0.03"; START="0.85"; SCHEDULE="late_linear";    LABEL="200k_w003_s085_lin" ;;
  4) STEP="200000"; WEIGHT="0.05"; START="0.90"; SCHEDULE="late_linear";    LABEL="200k_w005_s090_lin" ;;
  5) STEP="200000"; WEIGHT="0.03"; START="0.90"; SCHEDULE="late_quadratic"; LABEL="200k_w003_s090_quad" ;;
  *) echo "unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

CKPT="$PROJ/runs/omol25_4m_bgfm_dirft_250k/lightning_logs/version_12439949/checkpoints/midstep-step=${STEP}.ckpt"
OUT_DIR="$OUT_ROOT/$LABEL"
mkdir -p "$OUT_DIR"
cd "$PROJ"

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 2
fi

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
  "cell_4d_dirft_${LABEL}" \
  OMol25_4M \
  "BGFM_dirft_${LABEL}" \
  "$CKPT" \
  "$CONFIG" \
  "$EVAL_DATA" \
  "$N"
