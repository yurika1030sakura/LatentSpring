#!/bin/bash
#SBATCH -J ge_one
#SBATCH -p sapphire,serial_requeue
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=32 --mem=64G
#SBATCH -t 11:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_one-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/ge_one-%x-%j.err
#
# Single-arm CPU global-ensemble eval, so several arms can advance in parallel
# while the GPU partitions are saturated. Resumable: the logp stage flushes
# basin_logp_partial.json after every system.
# Usage: sbatch -J <tag> eval_global_ensemble_one.slurm <tag> <run_dir> <config> [n_systems]
# NOTE: absolute paths only -- SLURM copies this script to spool.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export PATH=$STORE/envs/flowmol/bin:$PATH
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
TAG=${1:?tag}; RUN=${2:?run dir under runs/abl}; CFGN=${3:?sweep config name}; NSYS=${4:-12}
NSTEPS=${5:-12}     # FFJORD ODE steps; vary to check discretisation robustness
SUB=${6:-}          # optional output-dir suffix so variants do not collide
SIGMA=${7:-0.15}    # perturbation std (A). Smaller -> better-conditioned IS
NPERT=${8:-8}       # perturbations per basin = importance samples per basin
ENS=$STORE/processed_data/ensembles
OUT=$STORE/runs/eval_ours/global_ensemble_cpu/${TAG}${SUB}
mkdir -p "$OUT"
cd "$REPO"
date
CKPT=$(ls -t "$STORE/runs/abl/$RUN"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "no ckpt for $TAG"; exit 0; }
echo "=== $TAG ckpt=$CKPT nsys=$NSYS"
# Always run: the logp stage resumes from basin_logp_partial.json and skips
# systems it already has, so re-running only costs the systems that are new.
# (No skip-gate on basin_logp.json -- that would block extending n_systems.)
"$FLOWMOL_PY" -u scripts/eval_global_ensemble.py logp \
  --checkpoint "$CKPT" --config "$REPO/configs/sweep/$CFGN.yaml" \
  --ensembles_dir "$ENS" --out_dir "$OUT" \
  --n_systems "$NSYS" --n_perturb "$NPERT" --sigma "$SIGMA" \
  --n_ode_steps "$NSTEPS" --batch_geoms 80 --device cpu --seed 0
LP="$OUT/basin_logp.json"; [ -s "$LP" ] || LP="$OUT/basin_logp_partial.json"
[ -s "$LP" ] || { echo "no logp output"; exit 0; }
OMP_NUM_THREADS=1 "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py analyze \
  --logp_json "$LP" --out_dir "$OUT" --kT_eV 1.0 --n_workers 16
echo "## GE ONE DONE $TAG ##"; date
