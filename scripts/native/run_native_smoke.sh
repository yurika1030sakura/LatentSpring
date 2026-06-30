#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/native/omol25_4m_bgfm_native_v1.yaml}
OUT=${2:-runs/native_smoke}
python scripts/run_train.py --config "$CFG" --fast --override_output_dir "$OUT"
