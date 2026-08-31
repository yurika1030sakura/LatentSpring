#!/bin/bash
#SBATCH -J ge_arm
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8 --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -t 6:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/gearm-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/gearm-%x-%j.err
#
# P1 (global / between-basin ensemble) eval for ONE arbitrary checkpoint tag,
# on the GPU partitions. CPU twin: eval_global_ensemble_one.slurm.
#
# Usage:
#   sbatch -J ge_<tag> scripts/fasrc/eval_global_ensemble_arm.slurm <label> <tag> [n_systems] [ode_steps] [suffix]
#     <label>  output directory name under runs/eval_ours/global_ensemble_gpu/
#     <tag>    directory name under runs/{p0,clean,abl}; configs/sweep/<tag>.yaml must exist
#
# Env overrides:  GE_DEVICE (cuda|cpu), GE_SIGMA, GE_NPERT, GE_ROOT
#
# Resumable: the logp stage flushes basin_logp_partial.json after every system
# and skips systems it already has, so preemption only costs the current system.
# NOTE: absolute paths only -- SLURM copies this script into its spool dir.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export PATH=$STORE/envs/flowmol/bin:$PATH
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
date

LABEL=${1:?output label, e.g. energy_s5}
TAG=${2:?checkpoint tag, e.g. a3_energy_only_s5}
NSYS=${3:-40}
NSTEPS=${4:-12}
SUB=${5:-}

RUNDIR=""
for root in "$STORE/runs/p0" "$STORE/runs/clean" "$STORE/runs/abl"; do
  [ -d "$root/$TAG" ] && { RUNDIR="$root/$TAG"; break; }
done
[ -z "$RUNDIR" ] && { echo "no run dir for $TAG"; exit 1; }
CFG="$REPO/configs/sweep/${TAG}.yaml"
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }

CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "no checkpoint for $TAG"; exit 1; }

ENS=$STORE/processed_data/ensembles
OUT=${GE_ROOT:-$STORE/runs/eval_ours/global_ensemble_gpu}/${LABEL}${SUB}
mkdir -p "$OUT"
echo "label=$LABEL tag=$TAG ckpt=$CKPT nsys=$NSYS steps=$NSTEPS out=$OUT"

"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad:
    sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
PY

"$FLOWMOL_PY" -u scripts/eval_global_ensemble.py logp \
  --checkpoint "$CKPT" --config "$CFG" \
  --ensembles_dir "$ENS" --out_dir "$OUT" \
  --n_systems "$NSYS" --n_perturb "${GE_NPERT:-8}" --sigma "${GE_SIGMA:-0.15}" \
  --n_ode_steps "$NSTEPS" --batch_geoms 80 --device "${GE_DEVICE:-cuda}" --seed 0

LP="$OUT/basin_logp.json"; [ -s "$LP" ] || LP="$OUT/basin_logp_partial.json"
[ -s "$LP" ] || { echo "no logp output"; exit 1; }
OMP_NUM_THREADS=1 "$FLOWMOL_PY" -u scripts/eval_global_ensemble.py analyze \
  --logp_json "$LP" --out_dir "$OUT" --kT_eV 1.0 --n_workers 8

echo "## GE ARM DONE $LABEL ##"; date
