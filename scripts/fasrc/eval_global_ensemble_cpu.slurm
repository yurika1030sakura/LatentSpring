#!/bin/bash
#SBATCH -J ge_cpu
#SBATCH -p sapphire,serial_requeue
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=32 --mem=64G
#SBATCH -t 11:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_cpu-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_cpu-%j.err
#
# CPU fallback for the P1 global-ensemble eval (the GPU partitions are saturated
# by the 30 P0 training jobs). Same code path, --device cpu, fewer systems.
# NOTE: absolute paths only -- SLURM copies this script to spool.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export PATH=$STORE/envs/flowmol/bin:$PATH
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
ENS=$STORE/processed_data/ensembles
ROOT=$STORE/runs/eval_ours/global_ensemble_cpu
mkdir -p "$ROOT"
cd "$REPO"
NSYS=${1:-10}
date

run_arm () {
  local TAG=$1 RUN=$2 CFGN=$3
  local CKPT
  CKPT=$(ls -t "$STORE/runs/abl/$RUN"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
  [ -z "$CKPT" ] && { echo "[ge-cpu] no ckpt for $TAG"; return 0; }
  echo "=== $TAG ckpt=$CKPT"
  local OUT=$ROOT/$TAG; mkdir -p "$OUT"
  if [ ! -s "$OUT/basin_logp.json" ]; then
    "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py logp \
      --checkpoint "$CKPT" --config "$REPO/configs/sweep/$CFGN.yaml" \
      --ensembles_dir "$ENS" --out_dir "$OUT" \
      --n_systems "$NSYS" --n_perturb 8 --sigma 0.15 \
      --n_ode_steps 12 --batch_geoms 80 --device cpu --seed 0
  fi
  OMP_NUM_THREADS=1 "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py analyze \
    --logp_json "$OUT/basin_logp.json" --out_dir "$OUT" --kT_eV 1.0 --n_workers 16
  echo "=== $TAG DONE"
}

run_arm energy_s2  a3_energy_only_s2  a3_energy_only_s2
run_arm control_s2 a1_fm_only_s2      a1_fm_only_s2
echo "## GE CPU DONE ##"; date
