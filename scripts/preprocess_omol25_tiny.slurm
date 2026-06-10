#!/bin/bash
# Tiny 100-molecule preprocess to validate the pipeline end-to-end BEFORE
# committing to the full 4M run. Writes to a separate tiny output dir.

#SBATCH -J omol25_tiny
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/tiny-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/tiny-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_prep"
OUT_DIR=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_tiny_processed
mkdir -p "$OUT_DIR"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/omol25"

export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "start: $(date)"
echo ""

python -u scripts/preprocess_omol25.py \
    --src /n/netscratch/ryl_lab/Lab/omol25/train_4M \
    --out_dir "$OUT_DIR" \
    --max_atoms 120 \
    --val_frac 0.1 \
    --max_n_mols 2000

python -u scripts/write_omol25_valencies.py --out_dir "$OUT_DIR"

echo ""
echo "end: $(date)"
echo "output:"
ls -lh "$OUT_DIR"
