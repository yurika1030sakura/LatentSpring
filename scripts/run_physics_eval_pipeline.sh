#!/bin/bash
# End-to-end physics-first eval for Paper 1 cells.
#
# Produces:
#   1. validity/connectivity CSV from scripts/evaluate_validity.py
#   2. generated sample JSON from scripts/export_samples_to_json.py
#   3. OMol25 raw/relax CSV from scripts/compute_omol25_energy.py
#   4. one-row stable schema from scripts/summarize_physics_eval.py
#
# Usage:
#   scripts/run_physics_eval_pipeline.sh \
#     cell_4d OMol25_4M BGFM \
#     /path/to/checkpoint.ckpt configs/omol25_4m_bgfm.yaml \
#     /n/netscratch/.../omol25_4m_processed 200
#
# Optional environment variables:
#   ANALYZER_DATA=<processed_dir>      # defaults to EVAL_DATA
#   OUT_DIR=<dir>                      # defaults to runs/eval/paper1
#   TABLE_CSV=<path>                   # defaults to $OUT_DIR/paper1_physics_table.csv
#   N_TIMESTEPS=100
#   BATCH_SIZE=16
#   FLOW_DEVICE=cuda
#   OMOL25_DEVICE=cpu
#   RELAX_STEPS=30
#   VANILLA=1                          # pass --vanilla to sampling/eval
#   NO_DISCRETE=0                      # allow discrete projection in validity eval
#   BGFM_GUIDANCE_WEIGHT=0.0           # late sampler score-guidance strength
#   BGFM_GUIDANCE_START=0.75
#   BGFM_GUIDANCE_SCHEDULE=late_linear
#   BGFM_GUIDANCE_CLIP=5.0
#   BGFM_GUIDANCE_RATIO=1.0
set -euo pipefail

# Keep cluster/user-site packages out of the conda envs.  Without this, a
# ~/.local NumPy 2.x install can shadow envs/flowmol's NumPy 1.x stack and
# break torch/wandb imports during evaluation.
export PYTHONNOUSERSITE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
ORIG_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [ "$#" -lt 7 ]; then
  echo "usage: $0 CELL_ID DATA_LABEL METHOD_LABEL CKPT CONFIG EVAL_DATA N_SAMPLES" >&2
  exit 2
fi

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
CELL_ID="$1"
DATA_LABEL="$2"
METHOD_LABEL="$3"
CKPT="$4"
CONFIG="$5"
EVAL_DATA="$6"
N="$7"

ANALYZER_DATA="${ANALYZER_DATA:-$EVAL_DATA}"
OUT_DIR="${OUT_DIR:-$PROJ/runs/eval/paper1}"
TABLE_CSV="${TABLE_CSV:-$OUT_DIR/paper1_physics_table.csv}"
N_TIMESTEPS="${N_TIMESTEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FLOW_DEVICE="${FLOW_DEVICE:-cuda}"
OMOL25_DEVICE="${OMOL25_DEVICE:-cpu}"
RELAX_STEPS="${RELAX_STEPS:-30}"
BGFM_GUIDANCE_WEIGHT="${BGFM_GUIDANCE_WEIGHT:-0.0}"
BGFM_GUIDANCE_START="${BGFM_GUIDANCE_START:-0.75}"
BGFM_GUIDANCE_SCHEDULE="${BGFM_GUIDANCE_SCHEDULE:-late_linear}"
BGFM_GUIDANCE_CLIP="${BGFM_GUIDANCE_CLIP:-5.0}"
BGFM_GUIDANCE_RATIO="${BGFM_GUIDANCE_RATIO:-1.0}"

TAG="${CELL_ID}_${DATA_LABEL}_${METHOD_LABEL}_n${N}"
VALIDITY_CSV="$OUT_DIR/${TAG}_validity.csv"
SAMPLES_JSON="$OUT_DIR/${TAG}_samples.json"
OMOL25_CSV="$OUT_DIR/${TAG}_omol25.csv"

mkdir -p "$OUT_DIR"
cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
# Some conda activate.d scripts assume this variable exists. Under set -u,
# activation can fail before evaluation starts unless we seed it.
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

activate_env() {
  export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
  conda activate "$1"
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ORIG_LD_LIBRARY_PATH"
}

VANILLA_ARG=""
if [ "${VANILLA:-0}" = "1" ]; then
  VANILLA_ARG="--vanilla"
fi

NO_DISCRETE_ARG=""
if [ "${NO_DISCRETE:-1}" = "1" ]; then
  NO_DISCRETE_ARG="--no-discrete"
fi

BGFM_GUIDANCE_ARGS=""
if python - <<PY
import sys
sys.exit(0 if float("$BGFM_GUIDANCE_WEIGHT") > 0.0 else 1)
PY
then
  BGFM_GUIDANCE_ARGS="--bgfm-guidance-weight $BGFM_GUIDANCE_WEIGHT --bgfm-guidance-start $BGFM_GUIDANCE_START --bgfm-guidance-schedule $BGFM_GUIDANCE_SCHEDULE --bgfm-guidance-clip $BGFM_GUIDANCE_CLIP --bgfm-guidance-ratio $BGFM_GUIDANCE_RATIO"
fi

echo "=== Paper 1 physics eval ==="
echo "cell      : $CELL_ID"
echo "data      : $DATA_LABEL"
echo "method    : $METHOD_LABEL"
echo "checkpoint: $CKPT"
echo "config    : $CONFIG"
echo "eval_data : $EVAL_DATA"
echo "out_dir   : $OUT_DIR"
echo ""

echo "=== Step 1: validity/connectivity -> $VALIDITY_CSV ==="
activate_env "$PROJ/envs/flowmol"
PYTHONPATH=. python scripts/evaluate_validity.py \
  --checkpoint "$CKPT" \
  --config "$CONFIG" \
  --eval_data "$EVAL_DATA" \
  --analyzer_data "$ANALYZER_DATA" \
  --n_samples "$N" \
  --n_timesteps "$N_TIMESTEPS" \
  --batch_size "$BATCH_SIZE" \
  --device "$FLOW_DEVICE" \
  $VANILLA_ARG \
  $NO_DISCRETE_ARG \
  $BGFM_GUIDANCE_ARGS \
  --out "$VALIDITY_CSV"

echo ""
echo "=== Step 2: export samples -> $SAMPLES_JSON ==="
PYTHONPATH=. python scripts/export_samples_to_json.py \
  --checkpoint "$CKPT" \
  --config "$CONFIG" \
  --eval_data "$EVAL_DATA" \
  --n_samples "$N" \
  --n_timesteps "$N_TIMESTEPS" \
  --batch_size "$BATCH_SIZE" \
  --device "$FLOW_DEVICE" \
  $VANILLA_ARG \
  $BGFM_GUIDANCE_ARGS \
  --out "$SAMPLES_JSON"
conda deactivate

echo ""
echo "=== Step 3: OMol25 raw/relax eval -> $OMOL25_CSV ==="
activate_env "$PROJ/envs/omol25"
python scripts/compute_omol25_energy.py \
  --samples_json "$SAMPLES_JSON" \
  --out "$OMOL25_CSV" \
  --relax_steps "$RELAX_STEPS" \
  --device "$OMOL25_DEVICE"
conda deactivate

echo ""
echo "=== Step 4: summary row -> $TABLE_CSV ==="
activate_env "$PROJ/envs/flowmol"
python scripts/summarize_physics_eval.py \
  --cell_id "$CELL_ID" \
  --data "$DATA_LABEL" \
  --method "$METHOD_LABEL" \
  --checkpoint "$CKPT" \
  --config "$CONFIG" \
  --eval_data "$EVAL_DATA" \
  --validity_csv "$VALIDITY_CSV" \
  --omol25_csv "$OMOL25_CSV" \
  --out "$TABLE_CSV" \
  --append
conda deactivate

echo ""
echo "DONE"
echo "  validity: $VALIDITY_CSV"
echo "  samples : $SAMPLES_JSON"
echo "  omol25  : $OMOL25_CSV"
echo "  table   : $TABLE_CSV"
