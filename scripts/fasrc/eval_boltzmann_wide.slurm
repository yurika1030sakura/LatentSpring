#!/bin/bash
#SBATCH -J boltz_wide
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/boltz_wide-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/boltz_wide-%x-%j.err
#
# WIDE Boltzmann eval: n_molecules 30 -> 120 to shrink the error bar ~2x.
# Purpose: the real-vs-shuffled (Y-scrambling) gap was Δr=+0.075 with SEM 0.073
# at n=30 (t=1.03, underpowered). At n=120 the SEM should fall to ~0.037, so the
# same gap would reach t~2.0. This decides whether the energy SIGNAL contributes
# beyond the generic variance-penalty regularization.
#
# Boltzmann half only (no de-novo sampling needed for this question).
# Usage: sbatch -J <tag> scripts/fasrc/eval_boltzmann_wide.slurm <ckpt> <config> <tag> [n_mol]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
CKPT=${1:?ckpt}; CFG=${2:?config}; TAG=${3:?tag}; NMOL=${4:-120}
PROC="$STORE/processed_data/omol25_4m_processed"
OUT="$STORE/runs/eval_ours/$TAG"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "host $(hostname)"; echo "tag=$TAG n_molecules=$NMOL"; date

# fail fast if the checkpoint is corrupt (learned this the hard way)
"$FLOWMOL_PY" - "$CKPT" <<'PY'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["state_dict"]
fp = [v for v in sd.values() if v.is_floating_point() and v.numel()]
bad = sum(int((~torch.isfinite(v)).sum()) for v in fp)
print(f"[ckpt-check] non-finite params: {bad}")
if bad: sys.exit(f"ABORT: checkpoint has {bad} non-finite params")
PY

echo "## stage1 FFJORD log p (n_molecules=$NMOL) ##"
"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_molecules "$NMOL" --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"

echo "## independent GFN2-xTB Boltzmann-R2 ##"
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"
echo "## WIDE EVAL DONE: $TAG ##"; cat "$OUT/boltz_independent.json"; date
