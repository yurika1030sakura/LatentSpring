#!/bin/bash
# Resume clean FM-seeded corrected BGFM from a Lightning checkpoint.
#
# This is used after walltime timeout of the original clean run. It restores
# trainer state from the latest clean BGFM checkpoint so mid-step checkpoints
# continue on the same global-step scale.
#
# Usage:
#   sbatch scripts/launch_omol25_bgfm_dirft_resume_a100.sh [ckpt]

#SBATCH -J omol25_bgfm_resume
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=256G
#SBATCH -t 2-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_bgfm_dirft_from_fm_resume/train-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_bgfm_dirft_from_fm_resume/train-%j.err
#SBATCH --signal=SIGUSR1@120

set -euo pipefail

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CONFIG="$PROJ/configs/omol25_4m_bgfm.yaml"
RESUME_CKPT="${1:-$PROJ/runs/omol25_4m_bgfm_dirft_from_fm/lightning_logs/version_13143711/checkpoints/midstep-step=150000.ckpt}"
OUT_DIR="$PROJ/runs/omol25_4m_bgfm_dirft_from_fm"

mkdir -p "$PROJ/runs/omol25_bgfm_dirft_from_fm_resume" "$OUT_DIR"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
conda activate "$PROJ/envs/flowmol"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
nvidia-smi -L
echo "config: $CONFIG"
echo "resume trainer checkpoint: $RESUME_CKPT"
echo "output_dir: $OUT_DIR"
echo "variant: resume clean FM -> corrected BGFM path-probe, cosine force loss"

python -u scripts/run_train.py \
  --config "$CONFIG" \
  --resume_ckpt_path "$RESUME_CKPT" \
  --override_output_dir "$OUT_DIR" \
  --override_base_lr 3.0e-5 \
  --override_max_epochs 4 \
  --override_bgfm_lambda_1 0.08 \
  --override_bgfm_force_loss_type cosine \
  --override_bgfm_probe_mode path \
  --override_bgfm_t_eval_values 0.92,0.97,0.985
