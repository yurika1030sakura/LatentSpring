#!/bin/bash
#SBATCH -J bgfm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_eval-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_eval-%j.err
#
# Full "ours" evaluation, filling Table 2 (de-novo on OMol25) and Table 3
# (Boltzmann fidelity). No teacher worker needed (sampling + FFJORD do not call
# the teacher). Args: $1=checkpoint  $2=config  $3=tag
#   Table 2: sample_to_json -> validity_from_json (validity+connected+PoseBusters
#            [config=mol all-checks, = Zatom-1]+uniqueness) + eval_xtb_relaxation (dE/atom)
#   Table 3: eval_boltzmann_stage1 (FFJORD log p) -> eval_boltzmann_independent
#            (GFN2-xTB R2 + ESS, circularity-free) [+ eSEN stage2 diagnostic]
set -eo pipefail
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
REPO=/home/renhaozhang_umass_edu/bgfm
FLOWMOL=$SCR/envs/flowmol/bin/python
OMOL=$SCR/envs/omol25/bin/python
ORACLE=$SCR/checkpoints/omol25/checkpoints/esen_sm_conserving_all.pt
CKPT=${1:?usage: eval_onpolicy.slurm <ckpt> <config> <tag> [processed_data_dir]}
CFG=${2:?config required}
TAG=${3:-ours}
PROC=${4:-$SCR/processed_data/omol25_50k}   # 4th arg: eval on the model's own split
OUT=$SCR/runs/eval_ours/$TAG; mkdir -p "$OUT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
export PATH=$SCR/envs/flowmol/bin:$PATH      # put the xtb binary on PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce FFJORD fragmentation OOM
cd "$REPO"
echo "host $(hostname); $(nvidia-smi -L 2>/dev/null)"; echo "ckpt=$CKPT tag=$TAG"; date

# ---------- Table 2: de-novo generation ----------
echo "## sample_to_json ##"; date
"$FLOWMOL" scripts/sample_to_json.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_samples 100 --n_timesteps 100 --batch_size 8 \
  --device cuda --out "$OUT/samples.json"
echo "## validity + PoseBusters + uniqueness ##"; date
"$FLOWMOL" benchmarks/omol25/validity_from_json.py \
  --samples_json "$OUT/samples.json" --out "$OUT/validity.json"
echo "## xTB relaxation dE ##"; date
"$FLOWMOL" scripts/eval_xtb_relaxation.py --samples "$OUT/samples.json" \
  --out_csv "$OUT/xtb.csv" --workers 4 2>&1 | tail -5 || echo "WARN xtb relax step failed"

# ---------- Table 3: Boltzmann fidelity ----------
echo "## stage1 FFJORD log p ##"; date
"$FLOWMOL" scripts/eval_boltzmann_stage1.py --checkpoint "$CKPT" --config "$CFG" \
  --eval_data "$PROC" --n_molecules 30 --n_perturb 8 --sigma 0.15 \
  --n_ode_steps 12 --n_hutchinson 2 --device cuda --out_dir "$OUT/boltz1"
echo "## independent GFN2-xTB Boltzmann-R2 + ESS (circularity-free) ##"; date
"$FLOWMOL" scripts/eval_boltzmann_independent.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_independent.csv" --out_json "$OUT/boltz_independent.json"
echo "## eSEN stage2 (in-loop diagnostic; circular, not for headline) ##"; date
"$OMOL" scripts/eval_boltzmann_stage2.py \
  --samples_json "$OUT/boltz1/boltzmann_samples.json" \
  --out_csv "$OUT/boltz_esen.csv" --ckpt "$ORACLE" --device cuda 2>&1 | tail -6 || echo "WARN eSEN stage2 failed"

echo "## EVAL DONE tag=$TAG ##"; ls -la "$OUT"; date
