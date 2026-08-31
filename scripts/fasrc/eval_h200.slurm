#!/bin/bash
#SBATCH -J bgfm_eval_h200
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_eval_h200-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/bgfm_eval_h200-%j.err
#
# Full "ours" evaluation -> fills Table 2 (de-novo on OMol25) and Table 3
# (Boltzmann fidelity) in notes/benchmark_plan.md. No teacher worker needed
# (sampling + FFJORD don't call the teacher). The independent GFN2-xTB metric
# (needs xtb on PATH) is the circularity-free headline; the eSEN stage-2 pass
# is an optional in-loop diagnostic that is skipped if the oracle isn't present.
#
# Usage: sbatch scripts/fasrc/eval_h200.slurm <ckpt> <config> <tag> [processed_data_dir]
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
CKPT=${1:?usage: eval_h200.slurm <ckpt> <config> <tag> [processed_data_dir]}
CFG=${2:?config required}
TAG=${3:-ours_4m}
PROC=${4:-$STORE/processed_data/omol25_4m_processed}
ORACLE="$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
OUT="$STORE/runs/eval_ours/$TAG"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH            # put the xtb binary on PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce FFJORD fragmentation OOM
cd "$REPO"
echo "host $(hostname); $(nvidia-smi -L 2>/dev/null)"; echo "ckpt=$CKPT tag=$TAG"; date

# ---------- Table 2: de-novo generation ----------
echo "## sample_to_json ##"; date
"$FLOWMOL_PY" scripts/sample_to_json.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_samples 100 --n_timesteps 100 --batch_size 8 \
  --device cuda --out "$OUT/samples.json"
echo "## validity + PoseBusters + uniqueness ##"; date
"$FLOWMOL_PY" benchmarks/omol25/validity_from_json.py \
  --samples_json "$OUT/samples.json" --out "$OUT/validity.json"
echo "## xTB relaxation dE ##"; date
"$FLOWMOL_PY" scripts/eval_xtb_relaxation.py --samples "$OUT/samples.json" \
  --out_csv "$OUT/xtb.csv" --workers 4 2>&1 | tail -5 || echo "WARN xtb relax step failed"

# ---------- Table 3: Boltzmann fidelity ----------
echo "## stage1 FFJORD log p ##"; date
"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_molecules 30 --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"
echo "## independent GFN2-xTB Boltzmann-R2 + ESS (circularity-free headline) ##"; date
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"

if [ -f "$ORACLE" ]; then
  echo "## eSEN stage2 (in-loop diagnostic; circular, not for headline) ##"; date
  "$OMOL25_PY" scripts/eval_boltzmann_stage2.py \
    --samples_json "$OUT/boltz1/boltzmann_samples.json" \
    --out_csv "$OUT/boltz_esen.csv" --ckpt "$ORACLE" --device cuda 2>&1 | tail -6 || echo "WARN eSEN stage2 failed"
else
  echo "## skipping eSEN stage2 diagnostic (oracle ckpt absent) ##"
fi

echo "## EVAL DONE tag=$TAG ##"; ls -la "$OUT"; date
