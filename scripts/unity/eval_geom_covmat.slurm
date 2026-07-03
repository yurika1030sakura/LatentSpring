#!/bin/bash
#SBATCH -J bgfm_covmat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_covmat-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/bgfm_covmat-%j.err
#
# Zero-shot GEOM COV/MAT -> Table 1 "ours". Args: $1=ckpt $2=config $3=partition(drugs|qm9) $4=tag
# Requires: the torsional-diffusion test split (test_mols.pkl) downloaded under
# $SCR/data/geom, and `datamol`+`loguru` installed in envs/flowmol.
set -eo pipefail
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
REPO=/home/renhaozhang_umass_edu/bgfm
FLOWMOL=$SCR/envs/flowmol/bin/python
CKPT=${1:?usage: eval_geom_covmat.slurm <ckpt> <config> <drugs|qm9> <tag>}
CFG=${2:?config required}
PART=${3:-drugs}
TAG=${4:-ours}
PDIR=$([ "$PART" = "drugs" ] && echo DRUGS || echo QM9)
TESTMOLS=$(find "$SCR/data/geom" -iname "test_mols.pkl" -ipath "*${PDIR}*" 2>/dev/null | head -1)
[ -z "$TESTMOLS" ] && TESTMOLS=$(find "$SCR/data/geom" -iname "test_mols.pkl" 2>/dev/null | head -1)
OUT=$SCR/runs/eval_ours/covmat_${PART}_${TAG}; mkdir -p "$OUT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$SCR/envs/flowmol/lib:${LD_LIBRARY_PATH:-}
cd "$REPO"
echo "host $(hostname); ckpt=$CKPT part=$PART tag=$TAG"; echo "test_mols=$TESTMOLS"; date
[ -z "$TESTMOLS" ] && { echo "ERROR: no test_mols.pkl under $SCR/data/geom (download not done?)"; exit 1; }
"$FLOWMOL" scripts/eval_geom_covmat.py --checkpoint "$CKPT" --config "$CFG" \
  --test_mols "$TESTMOLS" --partition "$PART" \
  --out_pkl "$OUT/generated_files.pkl" --out_json "$OUT/covmat.json" --device cuda
echo "## COVMAT DONE $PART $TAG ##"; cat "$OUT/covmat.json"; date
