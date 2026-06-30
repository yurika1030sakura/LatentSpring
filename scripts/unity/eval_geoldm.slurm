#!/bin/bash
#SBATCH -J eval_geoldm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/eval_geoldm-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/eval_geoldm-%j.err
# GeoLDM eval on OMol25: validity (sample 100 -> xyz2mol) + Boltzmann R2.
# Submit with --dependency=afterok:<stage2 jobid> so it runs automatically once
# the latent-diffusion stage (omol25_50k_geoldm_ldm) finishes. L40S so the 100-
# sample decode (up to 200-atom graphs) fits, matching the EDM eval.
set -uo pipefail
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
REPO=/home/renhaozhang_umass_edu/bgfm
EDM=$SCR/envs/edm/bin/python
OMOL=$SCR/envs/omol25/bin/python
ORACLE=$SCR/checkpoints/omol25/checkpoints/esen_sm_conserving_all.pt
PROC=$SCR/processed_data/omol25_50k
REFJSON=$SCR/runs/eval_real/bgfm_boltz1/boltzmann_samples.json
OUT=$SCR/runs/eval_real; mkdir -p "$OUT/geoldm_boltz1"
MODEL=$SCR/baselines/geoldm/outputs/omol25_50k_geoldm_ldm
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO/baselines/geoldm"

echo "######## GeoLDM : VALIDITY (sample 100 -> xyz2mol, unified path) ########"
$EDM -u sample_omol25_to_json.py --model_path "$MODEL" --datadir "$PROC" \
    --n_samples 100 --device cuda --out "$OUT/geoldm_samples.json"
$EDM -u "$REPO/benchmarks/omol25/validity_from_json.py" \
    --samples_json "$OUT/geoldm_samples.json" --out "$OUT/geoldm_validity_unified.json"

echo "######## GeoLDM : independent GFN2-xTB relaxation (fair cross-model metric) ########"
PATH="$SCR/envs/flowmol/bin:$PATH" $EDM -u "$REPO/scripts/eval_xtb_relaxation.py" \
    --samples "$OUT/geoldm_samples.json" --out_csv "$OUT/geoldm_xtb.csv" \
    --workers 8 --max_steps 200 || echo "[warn] xtb step failed"

echo "######## GeoLDM : BOLTZMANN stage1 (log p = -NLL, shared molecules) ########"
$EDM -u eval_omol25_boltzmann.py --model_path "$MODEL" --datadir "$PROC" \
    --ref_json "$REFJSON" --n_t_avg 16 --device cuda --out "$OUT/geoldm_boltz1/boltzmann_samples.json"

echo "######## GeoLDM : BOLTZMANN stage2 (oracle + R2) ########"
$OMOL "$REPO/scripts/eval_boltzmann_stage2.py" \
    --samples_json "$OUT/geoldm_boltz1/boltzmann_samples.json" \
    --out_csv "$OUT/geoldm_boltz2.csv" --ckpt "$ORACLE" --device cuda
echo "######## finalize: assemble COMPLETE 5-model fair table + publish ########"
cd "$REPO"
EVAL_OUT="$OUT" RESULTS_MD="$REPO/benchmarks/omol25/RESULTS.md" \
   python3 scripts/assemble_eval_table.py 2>&1 | tail -25
git add benchmarks/omol25/RESULTS.md 2>/dev/null && \
  git commit -m "benchmark(omol25): auto-publish fair-eval table (5-model complete, job ${SLURM_JOB_ID:-?})" 2>&1 | tail -2 || echo "(nothing to commit)"
git push origin norman 2>&1 | tail -2 || echo "[warn] push failed from compute node; committed locally, push later"
echo "######## GeoLDM EVAL DONE ########"; date
