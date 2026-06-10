#!/bin/bash
# Chained OMol25 evaluation: sample (flowmol env) + energy eval (omol25 env).
#
# Usage (internal): called by SLURM wrapper with args:
#   $1 = eval_data dir (e.g. data/qm9_processed)
#   $2 = tag (e.g. qm9, tmqm_organic, kraken)
#   $3 = n_samples
#   $4 = checkpoint path
set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
EVAL_DATA="$1"
TAG="$2"
N="$3"
CKPT="$4"

SAMPLES_JSON=$PROJ/runs/eval/v4_samples_${TAG}_n${N}.json
ENERGIES_CSV=$PROJ/runs/eval/v4_${TAG}_omol25_n${N}.csv

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh

# --- Step 1: sampling (flowmol env) ---
echo "=== Step 1: sample $N from $EVAL_DATA -> $SAMPLES_JSON ==="
conda activate "$PROJ/envs/flowmol"
PYTHONPATH=. python scripts/export_samples_to_json.py \
    --checkpoint "$CKPT" \
    --config configs/qm9_cfm.yaml \
    --eval_data "$EVAL_DATA" \
    --n_samples "$N" --batch_size 16 --vanilla \
    --out "$SAMPLES_JSON"
conda deactivate

# --- Step 2: OMol25 energy + BFGS relax (omol25 env) ---
echo ""; echo "=== Step 2: OMol25 energies -> $ENERGIES_CSV ==="
conda activate "$PROJ/envs/omol25"
python scripts/compute_omol25_energy.py \
    --samples_json "$SAMPLES_JSON" \
    --out "$ENERGIES_CSV" \
    --relax_steps 30 --device cpu
conda deactivate

echo ""; echo "DONE: $ENERGIES_CSV"
