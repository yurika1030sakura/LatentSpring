#!/bin/bash
#SBATCH -J bgfm_ood_boltz
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/eval_ood-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/eval_ood-%x-%j.err
#
# OOD Boltzmann-consistency eval (the "killer figure"): does the learned density
# rank HELD-OUT out-of-distribution geometries (transition metals / hypervalent /
# bulky-P) by their physical energy?
#
# Deliberately runs ONLY the Boltzmann half (stage1 FFJORD log p -> independent
# GFN2-xTB correlation). It does NOT do de-novo sampling: generating novel TM
# complexes is a different and much harder claim that we are not making, and the
# de-novo sampler NaNs on bond-carrying OOD data anyway.
#
# Usage: sbatch -J <tag> scripts/fasrc/eval_ood_boltzmann.slurm <ckpt> <config> <tag> <ood_processed_dir>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
CKPT=${1:?ckpt}; CFG=${2:?config}; TAG=${3:?tag}; PROC=${4:?ood processed dir}
OUT="$STORE/runs/eval_ours/$TAG"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH               # xtb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
echo "host $(hostname)"; nvidia-smi -L; echo "tag=$TAG  data=$PROC"; date

echo "## stage1: FFJORD log p on held-out OOD molecules ##"
"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_molecules 30 --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"

echo "## independent GFN2-xTB Boltzmann-R2 (circularity-free) ##"
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"

echo "## OOD BOLTZMANN EVAL DONE: $TAG ##"; cat "$OUT/boltz_independent.json"; date
