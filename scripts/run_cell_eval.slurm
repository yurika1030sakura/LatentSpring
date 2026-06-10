#!/bin/bash
# Uniform 5-cell ablation evaluation: chains Stage 1 (flowmol env, sample
# + connectivity + xyz2mol) and Stage 2 (omol25 env, OMol25 force/energy
# + BFGS relax).
#
# Usage:
#   sbatch scripts/run_cell_eval.slurm <cell_id> <ckpt> <config> [n_samples]
# Example:
#   sbatch scripts/run_cell_eval.slurm 4d \
#       runs/omol25_4m_bgfm/.../checkpoints/last.ckpt \
#       configs/omol25_4m_bgfm.yaml 500

#SBATCH -J cell_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/cell_eval-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/eval/cell_eval-%j.err

set -e

CELL_ID="${1:?usage: cell_id ckpt config [n_samples]}"
CKPT="${2:?usage: cell_id ckpt config [n_samples]}"
CONFIG="${3:?usage: cell_id ckpt config [n_samples]}"
N_SAMPLES="${4:-500}"

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
OUT_DIR=$PROJ/runs/eval/cell_${CELL_ID}
mkdir -p "$OUT_DIR" "$PROJ/runs/eval"
EVAL_DATA=$PROJ/data/qm9_processed   # for size distribution sampling

cd "$PROJ"

echo "host: $(hostname)"; nvidia-smi -L
echo "cell_id   : $CELL_ID"
echo "ckpt      : $CKPT"
echo "config    : $CONFIG"
echo "n_samples : $N_SAMPLES"
echo "out_dir   : $OUT_DIR"
echo ""

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

echo "=== Stage 1: sample + connectivity + xyz2mol (envs/flowmol) ==="
conda activate "$PROJ/envs/flowmol"
export PYTHONUNBUFFERED=1
python -u scripts/eval_cell_metrics.py \
    --cell_id "$CELL_ID" \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --eval_data "$EVAL_DATA" \
    --n_samples "$N_SAMPLES" \
    --out_dir "$OUT_DIR"
conda deactivate

echo ""
echo "=== Stage 2: OMol25 + BFGS relax (envs/omol25) ==="
conda activate "$PROJ/envs/omol25"
export PYTHONUNBUFFERED=1
python -u scripts/eval_cell_omol25_stage2.py \
    --cell_id "$CELL_ID" \
    --samples_json "$OUT_DIR/${CELL_ID}_samples.json" \
    --stage1_csv "$OUT_DIR/${CELL_ID}_stage1.csv" \
    --out_csv "$OUT_DIR/${CELL_ID}_eval.csv" \
    --device cuda \
    --relax_steps 50 \
    --fmax 0.01

echo ""
echo "=== Final eval CSV: $OUT_DIR/${CELL_ID}_eval.csv ==="
ls -lh "$OUT_DIR/"
