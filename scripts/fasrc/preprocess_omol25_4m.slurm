#!/bin/bash
#SBATCH -J omol25_4m_prep
#SBATCH -p sapphire,shared
#SBATCH -A woo_lab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/preprocess_4m-%j.out
#SBATCH -e /n/holylabs/woo_lab/Lab/yulili/bgfm/logs/preprocess_4m-%j.err
#
# OMol25 4M preprocessing (omol25 env / CPU). Reads DFT forces+energies straight
# from the raw LMDB and writes FlowMol-native .pt WITH a `forces` key (max_atoms=200)
# so the BGFM physics loss has force_1_true to train on.
#
# NOTE: the processed 4M set is ALREADY present at
#   $STORE/processed_data/omol25_4m_processed  (copied from the reference run).
# You only need this job to REGENERATE it (e.g. different max_atoms / subset).
# For a fast first pass add: --max_n_mols 1000000  (1M subset).
# Usage: sbatch scripts/fasrc/preprocess_omol25_4m.slurm
set -eo pipefail
source /n/home04/yulili/bgfm/scripts/fasrc/env.sh
RAW=/n/netscratch/ryl_lab/Lab/omol25/train_4M
OUT="$STORE/processed_data/omol25_4m_processed"
mkdir -p "$OUT"
echo "host: $(hostname)"; date
cd "$REPO"
"$OMOL25_PY" -u scripts/preprocess_omol25.py \
  --src "$RAW" --out_dir "$OUT" --max_atoms 200 --bond_tolerance 1.2
echo "## PREPROCESS DONE ##"; du -sh "$OUT"; date
