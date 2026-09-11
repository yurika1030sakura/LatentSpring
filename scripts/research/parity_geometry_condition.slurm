#!/bin/bash
#SBATCH -J bgfm-parity-xtb
#SBATCH -A ryl_lab
#SBATCH -p test
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/logs/%x-%j.err
set -euo pipefail
SOURCE_ROOT="${1:?immutable source required}"
RUN_ROOT="${2:?unique output required}"
CONDITION_INDEX="${3:?condition index required}"
STORE=/n/holylabs/woo_lab/Lab/yulili/bgfm
LAB_RUNS=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027/runs
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH="$SOURCE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$STORE/envs/flowmol/bin:$PATH" LD_LIBRARY_PATH="$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/bgfm_mpl_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/bgfm_cache_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$RUN_ROOT"
cd "$SOURCE_ROOT"
printf -v CONDITION_TAG '%02d' "$CONDITION_INDEX"
if [ "$CONDITION_INDEX" = 0 ]; then
  CONDITION_ROOT="$LAB_RUNS/parity_entropy_condition_00_v1"
else
  CONDITION_ROOT="$LAB_RUNS/parity_entropy_conditions_1_7_v1/condition_${CONDITION_TAG}"
fi
"$STORE/envs/flowmol/bin/python" - "$CONDITION_ROOT/panel.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if not p['complete'] or p['successful_arms'] != 6:
    raise ValueError('Require all six qualified training arms; keep failures in a separate assessment denominator')
PY
ARGS=(--samples base "$CONDITION_ROOT/condition_${CONDITION_TAG}_convex_s0/base_samples.pt")
for KIND in convex affine typed; do
  for REPLICA in 0 1; do
    ARGS+=(--samples "${KIND}_s${REPLICA}" "$CONDITION_ROOT/condition_${CONDITION_TAG}_${KIND}_s${REPLICA}/adapted_samples.pt")
  done
done
exec "$STORE/envs/flowmol/bin/python" -u scripts/research/assess_work_panel.py \
  "${ARGS[@]}" --out "$RUN_ROOT" --require-shared-parents \
  --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
