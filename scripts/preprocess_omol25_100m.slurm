#!/bin/bash
# Plan C: Preprocess OMol25 100M (full training set) into FlowMol3-native
# tensor files with BGFM labels (forces + energies). Output goes to
# netscratch (expected ~525-600 GB processed).
#
# ETA: ~12 hours based on 4M preprocess (48 min at ~1400 mol/s).

#SBATCH -J omol25_100m_prep
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/prep100m-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/prep100m-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_prep"
OUT_DIR=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_100m_processed
mkdir -p "$OUT_DIR"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/omol25"

export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "start: $(date)"
echo "disk free on netscratch:"
df -h "$OUT_DIR" | tail -1
echo ""

python -u scripts/preprocess_omol25.py \
    --src /n/netscratch/ryl_lab/Lab/omol25_100m/train \
    --out_dir "$OUT_DIR" \
    --max_atoms 200 \
    --val_frac 0.005 \
    --shard_size 5000000

python -u scripts/write_omol25_valencies.py --out_dir "$OUT_DIR"

echo ""
echo "end: $(date)"
ls -lh "$OUT_DIR"
du -sh "$OUT_DIR"
