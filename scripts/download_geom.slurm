#!/bin/bash
# Download + process GEOM-Drugs for the ICLR paper (E1 OOD training set).
#
# GEOM-Drugs raw is ~40 GB. Processing produces three pickle files
# (train/test/val) and then FlowMol3's process_geom.py converts each to
# its .pt format in data/geom_5_kekulized/.
#
# Uses the MiDi-preprocessed GEOM archive hosted at bits.csb.pitt.edu
# (per flowmol3/readme.md section "GEOM-Drugs"). Download is multi-file
# wget recursive; processing is CPU-heavy (RDKit).
#
# Total wall time estimate:
#   - Download: 30-60 min depending on network (~40 GB)
#   - Process train (~2M conformers): 4-8 hours
#   - Process test (~0.2M): 30-60 min
#   - Process val (~0.2M): 30-60 min
# Walltime set to 24h for safety.

#SBATCH -J cfm_geom
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/geom-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/geom-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
FLOWMOL=$PROJ/baselines/flowmol3

echo "host: $(hostname)"
echo "start: $(date)"
echo ""

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

# ---------------------------------------------------------------------------
# Step 1: download raw GEOM-Drugs (skip if already present).
# ---------------------------------------------------------------------------
echo "=== Step 1: download GEOM-Drugs raw ==="
cd "$FLOWMOL"
RAW_DIR=data/geom_raw
if [ -f "$RAW_DIR/train_data.pickle" ] && [ -f "$RAW_DIR/test_data.pickle" ] && [ -f "$RAW_DIR/val_data.pickle" ]; then
    echo "raw GEOM pickles already present in $RAW_DIR; skipping download."
else
    mkdir -p "$RAW_DIR"
    wget -r -np -nH --cut-dirs=2 --reject "index.html*" \
        -P "$RAW_DIR" https://bits.csb.pitt.edu/files/geom_raw/
    echo "download done: $(date)"
fi
ls -lh "$RAW_DIR/"

# ---------------------------------------------------------------------------
# Step 2: process each split. FlowMol3's process_geom.py takes split file +
#         config. Use geom_5_kekulized config (atom map C/H/N/O/F/P/S/Cl/Br/I,
#         no explicit aromaticity).
# ---------------------------------------------------------------------------
GEOM_CONFIG=configs/configs_dataprocessing/geom_5_kekulized.yaml
OUT_DIR=data/geom_5_kekulized

for split in train val test; do
    echo ""
    echo "=== processing $split split ==="
    SPLIT_FILE=$RAW_DIR/${split}_data.pickle
    OUT_FILE=$OUT_DIR/${split}_data_processed.pt
    if [ -f "$OUT_FILE" ]; then
        echo "$OUT_FILE already exists; skipping."
        continue
    fi
    # Serial to avoid /dev/shm exhaustion that killed prior runs; QM9
    # hit the same issue. ~10-20 h for the 2M-conformer train split.
    python process_geom.py "$SPLIT_FILE" \
        --config "$GEOM_CONFIG" \
        --n_cpus 1 \
        --chunk_size 50
    echo "$split done: $(date)"
done

echo ""
echo "=== summary ==="
ls -lh "$OUT_DIR/"
echo ""
echo "ALL DONE: $(date)"
