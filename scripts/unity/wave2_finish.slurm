#!/bin/bash
#SBATCH -J wave2_finish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/wave2_finish-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/wave2_finish-%j.err
# Finish the fair-eval pass as a BATCH job (the interactive-node nohup version was
# killed by a session teardown). Samples + unified validity already done; this does:
#   (2) GFN2-xTB relaxation on BGFM/FM samples
#   (3a) stage2 re-run (+Spearman) for fm_only/edm/symphony (reuse existing boltz1)
#   (3b) de-noised FFJORD (n_hutch 16, n_ode 20) BGFM/FM -> stage2 -> *_boltz2_ln.csv
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

echo "#### (2) xtb relaxation on BGFM/FM ####"
for m in bgfm fm_only; do
  echo "-- xtb $m --"
  PATH="$SCR/envs/flowmol/bin:$PATH" $FLOWMOL scripts/eval_xtb_relaxation.py \
     --samples "$OUT/${m}_samples.json" --out_csv "$OUT/${m}_xtb.csv" \
     --workers 8 --max_steps 200 2>&1 | tail -2
done

echo "#### (3a) stage2 (+Spearman) for fm_only/edm/symphony (reuse boltz1) ####"
for m in fm_only edm symphony; do
  [ -f "$OUT/${m}_boltz1/boltzmann_samples.json" ] || continue
  echo "-- stage2 $m --"
  $OMOL scripts/eval_boltzmann_stage2.py --samples_json "$OUT/${m}_boltz1/boltzmann_samples.json" \
     --out_csv "$OUT/${m}_boltz2.csv" --ckpt "$ORACLE" --device cuda 2>&1 | grep -iE "Pearson|Spearman|groups" | tail -3
done

echo "#### (3b) de-noised FFJORD (n_hutch 16, n_ode 20) BGFM/FM ####"
for m in bgfm fm_only; do
  rm -rf "$OUT/${m}_boltz1_ln"; mkdir -p "$OUT/${m}_boltz1_ln"
  $FLOWMOL scripts/eval_boltzmann_stage1.py --checkpoint "${CKPT[$m]}" --config "$SCR/omol25_50k_$m.yaml" \
     --eval_data "$PROC" --n_molecules 50 --n_perturb 16 --sigma 0.15 --max_atoms_filter 50 \
     --n_ode_steps 20 --n_hutchinson 16 --device cuda --out_dir "$OUT/${m}_boltz1_ln" 2>&1 | tail -3
  $OMOL scripts/eval_boltzmann_stage2.py --samples_json "$OUT/${m}_boltz1_ln/boltzmann_samples.json" \
     --out_csv "$OUT/${m}_boltz2_ln.csv" --ckpt "$ORACLE" --device cuda 2>&1 | grep -iE "Pearson|Spearman|POOLED|groups" | tail -5
done
echo "#### WAVE2_FINISH DONE ####"; date
