#!/bin/bash
#SBATCH -J p6_arm
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:1 --mem=48G
#SBATCH -t 6:00:00
#SBATCH --requeue
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6arm-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p6arm-%x-%j.err
#
# P6 (likelihood-estimator validation) for ONE arbitrary checkpoint tag.
# Generalises validate_estimator.slurm, which hard-codes a3_energy_only_s2.
#
# Usage:
#   sbatch -J p6_<tag> scripts/fasrc/validate_estimator_arm.slurm <tag> [out_suffix]
#     <tag>  directory name under runs/{p0,clean,abl}; configs/sweep/<tag>.yaml must exist
#
# Env overrides (so the same script serves the GPU and the CPU-insurance runs):
#   P6_DEVICE      cuda | cpu                (default cuda)
#   P6_NMOL P6_MINAT P6_MAXAT
#   P6_STEPS       comma list of ODE steps   (default 4,8,12,24,48)
#   P6_PROBES      comma list of probes      (default 1,2,4,8)
#   P6_NREP        repeats per cell          (default 3)
#   P6_EXACT_MOLS  0 disables the exact-divergence cross-check
#
# Resumable by construction: one JSON per (steps,probes,repeat) cell, existing
# cells are skipped, so a gpu_requeue preemption just resumes.
# NOTE: absolute paths only -- SLURM copies this script into its spool dir.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cd "$REPO"
date

TAG=${1:?checkpoint tag, e.g. a3_energy_only_s3}
SUB=${2:-}

RUNDIR=""
for root in "$STORE/runs/p0" "$STORE/runs/clean" "$STORE/runs/abl"; do
  [ -d "$root/$TAG" ] && { RUNDIR="$root/$TAG"; break; }
done
[ -z "$RUNDIR" ] && { echo "no run dir for $TAG under runs/{p0,clean,abl}"; exit 1; }
CFG="$REPO/configs/sweep/${TAG}.yaml"
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }

CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/midstep-step=30000.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && { echo "no checkpoint for $TAG"; exit 1; }

OUT="$STORE/runs/eval_ours/p6_estimator/${TAG}${SUB}"
mkdir -p "$OUT"
echo "tag=$TAG ckpt=$CKPT out=$OUT device=${P6_DEVICE:-cuda}"

# Weight-finiteness gate (we once drew conclusions from a NaN checkpoint).
"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad:
    sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
PY

EXACT_ARGS="--exact_max_atoms 15 --exact_n_mols ${P6_EXACT_MOLS:-3} --exact_n_geom 3 --exact_ode_steps 12 --exact_hutch_reps 6"
if [ "${P6_EXACT_MOLS:-3}" = "0" ]; then EXACT_ARGS="--exact_n_mols 0"; fi

"$FLOWMOL_PY" -u scripts/validate_likelihood_estimator.py \
  --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" --out_dir "$OUT" \
  --n_molecules "${P6_NMOL:-12}" --n_perturb 8 --sigma 0.15 \
  --min_atoms "${P6_MINAT:-8}" --max_atoms "${P6_MAXAT:-40}" \
  --ode_steps "${P6_STEPS:-4,8,12,24,48}" --probes "${P6_PROBES:-1,2,4,8}" \
  --crn_cells 12:2 --n_repeat "${P6_NREP:-3}" \
  $EXACT_ARGS \
  --device "${P6_DEVICE:-cuda}" --seed 0

# canonical (with-reference) report, then a drop-reference variant alongside it
"$FLOWMOL_PY" -u scripts/validate_likelihood_estimator.py --analyze_only \
  --out_dir "$OUT" > "$OUT/report.txt" 2>&1 || true
"$FLOWMOL_PY" -u scripts/validate_likelihood_estimator.py --analyze_only --drop_ref \
  --out_dir "$OUT" > "$OUT/report_dropref.txt" 2>&1 || true
mv "$OUT/report.json" "$OUT/report_dropref.json" 2>/dev/null || true
"$FLOWMOL_PY" -u scripts/validate_likelihood_estimator.py --analyze_only \
  --out_dir "$OUT" > /dev/null 2>&1 || true

echo "## P6 ESTIMATOR DONE $TAG ##"; date
