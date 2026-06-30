#!/bin/bash
#SBATCH -J wave2_finish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/wave2_finish-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/wave2_finish-%j.err
# FAIR-eval finish (v2, post fairness-audit). Batch job (teardown-safe). Fixes:
#  - #4 sample BGFM/FM with --vanilla (no patch_flowmol steric cleanup) so the
#    cross-family validity/xTB use raw geometries like EDM/Symphony.
#  - #7 per-atom xTB ΔE; recompute xTB for ALL 4 ready models (per-atom field).
#  - #1 validity over n=100 attempts (assembler picks the larger-n file).
#  - #11/#13 Spearman + de-noised FFJORD (n_hutch16/ode20) for BGFM/FM.
# Boltzmann (boltz1) is on held-out molecules, unaffected by the --vanilla resample.
set -uo pipefail
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
REPO=/home/renhaozhang_umass_edu/bgfm
FLOWMOL=$SCR/envs/flowmol/bin/python
OMOL=$SCR/envs/omol25/bin/python
ORACLE=$SCR/checkpoints/omol25/checkpoints/esen_sm_conserving_all.pt
PROC=$SCR/processed_data/omol25_50k
OUT=$SCR/runs/eval_real
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
echo "host $(hostname); $(nvidia-smi -L)"; date
cd "$REPO"
declare -A CKPT=(
  [bgfm]="$SCR/runs/omol25_50k_bgfm/lightning_logs/version_5/checkpoints/epoch=14-step=26340.ckpt"
  [fm_only]="$SCR/runs/omol25_50k_fm_only/lightning_logs/version_1/checkpoints/epoch=14-step=26340.ckpt"
)

echo "#### (0) re-sample BGFM/FM with --vanilla (raw geometries, fair vs baselines) ####"
for m in bgfm fm_only; do
  $FLOWMOL scripts/sample_to_json.py --checkpoint "${CKPT[$m]}" --config "$SCR/omol25_50k_$m.yaml" \
     --eval_data "$PROC" --n_samples 100 --n_timesteps 100 --batch_size 8 --vanilla \
     --device cuda --out "$OUT/${m}_samples.json" 2>&1 | tail -3
done

echo "#### (1) unified xyz2mol validity over n=100 attempts (bgfm/fm/edm; symphony already n=100) ####"
for m in bgfm fm_only edm; do
  [ -f "$OUT/${m}_samples.json" ] || continue
  $FLOWMOL "$REPO/benchmarks/omol25/validity_from_json.py" \
     --samples_json "$OUT/${m}_samples.json" --out "$OUT/${m}_validity_unified.json" 2>&1 | tail -1
done

echo "#### (2) GFN2-xTB relaxation (per-atom ΔE) on ALL 4 ready models ####"
for m in bgfm fm_only edm symphony; do
  [ -f "$OUT/${m}_samples.json" ] || continue
  echo "-- xtb $m --"
  PATH="$SCR/envs/flowmol/bin:$PATH" $FLOWMOL scripts/eval_xtb_relaxation.py \
     --samples "$OUT/${m}_samples.json" --out_csv "$OUT/${m}_xtb.csv" \
     --workers 8 --max_steps 200 2>&1 | tail -2
done

echo "#### (3) stage2 (+Spearman) for all 4 (reuse existing boltz1) ####"
for m in bgfm fm_only edm symphony; do
  [ -f "$OUT/${m}_boltz1/boltzmann_samples.json" ] || continue
  echo "-- stage2 $m --"
  $OMOL scripts/eval_boltzmann_stage2.py --samples_json "$OUT/${m}_boltz1/boltzmann_samples.json" \
     --out_csv "$OUT/${m}_boltz2.csv" --ckpt "$ORACLE" --device cuda 2>&1 | grep -iE "Pearson|Spearman|groups" | tail -3
done

echo "#### (4) de-noised FFJORD (n_hutch 16, n_ode 20) BGFM/FM ####"
for m in bgfm fm_only; do
  rm -rf "$OUT/${m}_boltz1_ln"; mkdir -p "$OUT/${m}_boltz1_ln"
  $FLOWMOL scripts/eval_boltzmann_stage1.py --checkpoint "${CKPT[$m]}" --config "$SCR/omol25_50k_$m.yaml" \
     --eval_data "$PROC" --n_molecules 50 --n_perturb 16 --sigma 0.15 --max_atoms_filter 50 \
     --n_ode_steps 20 --n_hutchinson 16 --device cuda --out_dir "$OUT/${m}_boltz1_ln" 2>&1 | tail -3
  $OMOL scripts/eval_boltzmann_stage2.py --samples_json "$OUT/${m}_boltz1_ln/boltzmann_samples.json" \
     --out_csv "$OUT/${m}_boltz2_ln.csv" --ckpt "$ORACLE" --device cuda 2>&1 | grep -iE "Pearson|Spearman|groups" | tail -3
done

echo "#### finalize: assemble fair table + publish to norman ####"
cd "$REPO"
EVAL_OUT="$OUT" RESULTS_MD="$REPO/benchmarks/omol25/RESULTS.md" \
   python3 scripts/assemble_eval_table.py 2>&1 | tail -30
git add benchmarks/omol25/RESULTS.md 2>/dev/null && \
  git commit -m "benchmark(omol25): auto-publish fair-eval table v2 (audit-fixed, job ${SLURM_JOB_ID:-?})" 2>&1 | tail -2 || echo "(nothing to commit)"
git push origin norman 2>&1 | tail -2 || echo "[warn] push failed from compute node; committed locally, push later"
echo "#### WAVE2_FINISH DONE ####"; date
