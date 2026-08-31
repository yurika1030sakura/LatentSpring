#!/bin/bash
#SBATCH -J bgfm_eval_sweep
#SBATCH -p gpu_requeue,gpu_h200
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/eval_sweep-%x-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/eval_sweep-%x-%j.err
#
# Auto-eval one sweep run: discovers the newest last.ckpt under the run's
# output_dir and runs the Table-2 (de-novo) + Table-3 (Boltzmann) pipeline.
# Meant to be chained: sbatch --dependency=afterany:<trainjob> ... <name>.
# Exits 0 gracefully if the training run produced no checkpoint.
# Usage: sbatch -J <name> scripts/fasrc/eval_sweep.slurm <name>
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
NAME="${1:?usage: sbatch scripts/fasrc/eval_sweep.slurm <sweep config name>}"
CFG="$REPO/configs/sweep/${NAME}.yaml"
RUNDIR="$STORE/runs/sweep/${NAME}"
ORACLE="$STORE/checkpoints/omol25/esen_sm_conserving_all.pt"
OUT="$STORE/runs/eval_sweep/${NAME}"; mkdir -p "$OUT"
export LD_LIBRARY_PATH=$STORE/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$STORE/envs/flowmol/bin:$PATH            # xtb on PATH
echo "host $(hostname); $(nvidia-smi -L 2>/dev/null)"; echo "run=$NAME"; date

# --- discover newest checkpoint (last.ckpt preferred, else any .ckpt) ---
CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1)
[ -z "$CKPT" ] && CKPT=$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "NO CHECKPOINT under $RUNDIR -- training produced none; nothing to eval. Exiting 0."
  exit 0
fi
echo "checkpoint: $CKPT"

PROC="$STORE/processed_data/omol25_4m_processed"
cd "$REPO"

# ---------- Table 2: de-novo generation ----------
echo "## sample_to_json ##"; date
"$FLOWMOL_PY" scripts/sample_to_json.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_samples 100 --n_timesteps 100 --batch_size 8 \
  --device cuda --out "$OUT/samples.json"
echo "## validity + PoseBusters + uniqueness ##"; date
"$FLOWMOL_PY" benchmarks/omol25/validity_from_json.py \
  --samples_json "$OUT/samples.json" --out "$OUT/validity.json" || echo "WARN validity step failed"
echo "## xTB relaxation dE ##"; date
"$FLOWMOL_PY" scripts/eval_xtb_relaxation.py --samples "$OUT/samples.json" \
  --out_csv "$OUT/xtb.csv" --workers 4 2>&1 | tail -5 || echo "WARN xtb step failed"

# ---------- Table 3: Boltzmann fidelity (independent GFN2-xTB = headline) ----------
echo "## stage1 FFJORD log p ##"; date
"$FLOWMOL_PY" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_molecules 30 --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"
echo "## independent GFN2-xTB Boltzmann-R2 + ESS ##"; date
"$FLOWMOL_PY" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json" || echo "WARN indep boltz failed"

if [ -f "$ORACLE" ]; then
  echo "## eSEN stage2 (in-loop diagnostic) ##"; date
  "$OMOL25_PY" scripts/eval_boltzmann_stage2.py \
    --samples_json "$OUT/boltz1/boltzmann_samples.json" \
    --out_csv "$OUT/boltz_esen.csv" --ckpt "$ORACLE" --device cuda 2>&1 | tail -6 || echo "WARN eSEN stage2 failed"
fi
echo "## EVAL DONE $NAME ##"; ls -la "$OUT"; date
