#!/bin/bash
#SBATCH -J omol25_4m_prep
#SBATCH --partition=cpu
#SBATCH --account=pi_bsilva_umass_edu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/preprocess_4m-%j.out
#SBATCH -e /home/renhaozhang_umass_edu/scratch_workspace/bgfm/logs/preprocess_4m-%j.err
#
# OMol25 4M preprocessing (omol25 env). Extracts all 160 LMDB shards then builds
# the FlowMol-native .pt WITH DFT forces + energies (max_atoms=200) so the BGFM
# physics loss actually has force_1_true to train on. No GPU (reads DFT labels
# straight from LMDB, no NN eval).
set -eo pipefail
SCR=/home/renhaozhang_umass_edu/scratch_workspace/bgfm
REPO=/home/renhaozhang_umass_edu/bgfm
RAW=$SCR/data/omol25/train_4M
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
cd "$REPO"
echo "host $(hostname)"; date

# 1. extract all shards if not already present
N=$(ls "$RAW"/*.aselmdb 2>/dev/null | wc -l)
echo "shards present: $N / 160"
if [ "$N" -lt 160 ]; then
  echo "extracting train_4M.tar.gz ..."; date
  tar xzf "$SCR/data/omol25/train_4M.tar.gz" -C "$SCR/data/omol25/"
  echo "shards now: $(ls "$RAW"/*.aselmdb 2>/dev/null | wc -l)"; date
fi

# 2. preprocess -> processed_data/omol25_4m (positions/atom_types/charges/FORCES/energies)
"$SCR/envs/omol25/bin/python" -u scripts/preprocess_omol25.py \
  --src "$RAW" \
  --out_dir "$SCR/processed_data/omol25_4m" \
  --max_atoms 200 --bond_tolerance 1.2
echo "## PREPROCESS DONE ##"; du -sh "$SCR/processed_data/omol25_4m"; date
