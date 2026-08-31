#!/bin/bash
#SBATCH -J p3ctrl
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p3ctrl-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/p3ctrl-%x-%j.err
#
# P3: controlled-perturbation Boltzmann eval (kills the geometry-distance confound).
#
# Usage:
#   sbatch scripts/fasrc/eval_controlled.slurm <ckpt_tag> <perturb_mode> \
#          [n_mol] [n_perturb] [scales] [extra flags...]
#
#   <ckpt_tag>     directory name under runs/{p0,clean,abl} whose newest finite
#                  checkpoint is evaluated, e.g. a3_energy_only_s2
#   <perturb_mode> gaussian | fixed_rmsd | torsion | bond_angle | normal_mode
#   <scales>       comma-separated; interpreted as --rmsd for fixed_rmsd /
#                  torsion / bond_angle / normal_mode, and as --sigma for gaussian
#
# Examples:
#   sbatch -J e_fx scripts/fasrc/eval_controlled.slurm a3_energy_only_s2 fixed_rmsd 20 8 0.15
#   sbatch -J e_ml scripts/fasrc/eval_controlled.slurm a3_energy_only_s2 fixed_rmsd 60 8 0.10,0.20,0.40
#   sbatch -J e_to scripts/fasrc/eval_controlled.slurm a3_energy_only_s2 torsion    60 8 0.15
#
# NOTE (hard-won): never use $0 / dirname $0 in a SLURM script -- SLURM copies the
# script into its spool dir, so relative resolution silently points at the wrong
# tree. Absolute paths only.
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh

TAG=${1:?ckpt tag, e.g. a3_energy_only_s2}
MODE=${2:-fixed_rmsd}
NMOL=${3:-60}
NPERT=${4:-8}
SCALES=${5:-0.15}
shift 5 2>/dev/null || shift $#
EXTRA="$@"

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

export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

# scale flag: gaussian is parameterised by sigma (A), everything else by target RMSD
if [ "$MODE" = "gaussian" ]; then SCALE_FLAG="--sigma $SCALES"; else SCALE_FLAG="--rmsd $SCALES"; fi

SAFE_SCALES=$(echo "$SCALES" | tr ',' '_')
# P3_SUFFIX keeps re-runs at different fidelity (CPU 8-step vs GPU 12-step, or a
# different --max_atoms_filter) from silently overwriting each other's results.
OUT="$STORE/runs/eval_p3/${TAG}__${MODE}__${SAFE_SCALES}${P3_SUFFIX:-}"
mkdir -p "$OUT"
echo "tag=$TAG mode=$MODE scales=$SCALES n_mol=$NMOL n_pert=$NPERT"
echo "ckpt=$CKPT"; echo "out=$OUT"; date

# Weight-finiteness gate (we have previously drawn conclusions from a NaN ckpt).
"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
bad = sum(int((~torch.isfinite(v)).sum()) for v in sd.values() if v.is_floating_point())
print(f"[ckpt-check] non-finite params: {bad}")
if bad:
    sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
PY

# 1. controlled perturbations + log p_theta  (drops the reference conformer: it is
#    a leverage point AND has RMSD 0, which would break the fixed-RMSD control)
"$FLOWMOL_PY" scripts/eval_boltzmann_controlled.py \
  --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$STORE/processed_data/omol25_4m_processed" \
  --perturb_mode "$MODE" $SCALE_FLAG \
  --n_molecules "$NMOL" --n_perturb "$NPERT" --drop_reference \
  --n_ode_steps "${P3_ODE_STEPS:-12}" --n_hutchinson "${P3_HUTCH:-2}" \
  --device "${P3_DEVICE:-cuda}" \
  --max_atoms_filter "${P3_MAX_ATOMS:-60}" --out_dir "$OUT/boltz1" $EXTRA

# 2. independent GFN2-xTB energies + per-group correlation (same script the
#    headline table uses, so the numbers are directly comparable)
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz.csv" --out_json "$OUT/boltz.json"

# 3. RMSD-controlled statistics (partial correlation + stratification)
"$FLOWMOL_PY" scripts/analyze_partial_correlation.py \
  --records_csv "$OUT/boltz_records.csv" \
  --meta_csv "$OUT/boltz1/perturbation_meta.csv" \
  --label "${TAG}__${MODE}__${SAFE_SCALES}" \
  --out_json "$OUT/partial_corr.json" --out_csv "$OUT/partial_corr_groups.csv"

echo "## P3 CONTROLLED EVAL DONE $TAG $MODE ##"
cat "$OUT/boltz.json"
date
