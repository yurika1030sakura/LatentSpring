#!/bin/bash
#SBATCH -J ge_eval
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=4 --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_eval-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_eval-%j.err
#
# P1 global-ensemble eval, step 2+3: FFJORD log p on every basin representative
# (GPU) then xTB energies + basin-mass metrics (CPU), for one energy-arm and one
# control-arm checkpoint per seed.
# NOTE: absolute paths only -- SLURM copies this script to spool, so $0 is useless.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export PATH=$STORE/envs/flowmol/bin:$PATH
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENS=$STORE/processed_data/ensembles
ROOT=$STORE/runs/eval_ours/global_ensemble
mkdir -p "$ROOT"
cd "$REPO"
NSYS=${1:-60}
date; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_arm () {   # $1 = tag, $2 = run dir under runs/abl, $3 = sweep config name
  local TAG=$1 RUN=$2 CFGN=$3
  local CKPT
  CKPT=$(ls -t "$STORE/runs/abl/$RUN"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "[ge] no ckpt for $TAG ($RUN) -- skip"; return 0; fi
  echo "=== $TAG  ckpt=$CKPT"
  local OUT=$ROOT/$TAG
  mkdir -p "$OUT"
  # checkpoint finiteness gate (lesson: we once evaluated a NaN checkpoint)
  "$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad: sys.exit(f"ABORT: {bad} non-finite params")
PY
  if [ ! -s "$OUT/basin_logp.json" ]; then
    "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py logp \
      --checkpoint "$CKPT" --config "$REPO/configs/sweep/$CFGN.yaml" \
      --ensembles_dir "$ENS" --out_dir "$OUT" \
      --n_systems "$NSYS" --n_perturb 8 --sigma 0.15 \
      --n_ode_steps 12 --batch_geoms 80 --device cuda --seed 0
  else
    echo "[ge] reusing existing $OUT/basin_logp.json"
  fi
  # If the job was preempted mid-arm, analyse whatever systems completed.
  local LP="$OUT/basin_logp.json"
  [ -s "$LP" ] || LP="$OUT/basin_logp_partial.json"
  [ -s "$LP" ] || { echo "[ge] no logp output for $TAG"; return 0; }
  "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py analyze \
    --logp_json "$LP" --out_dir "$OUT" --kT_eV 1.0 --n_workers 4
  echo "=== $TAG DONE"
}

run_arm energy_s2  a3_energy_only_s2  a3_energy_only_s2
run_arm control_s2 a1_fm_only_s2      a1_fm_only_s2
run_arm energy_s3  a3_energy_only_s3  a3_energy_only_s3
run_arm control_s3 a1_fm_only_s3      a1_fm_only_s3

echo "## GLOBAL ENSEMBLE EVAL DONE ##"; date
