#!/bin/bash
# Preprocess OMol25 4M LMDB -> FlowMol3 native tensor files. Runs in omol25
# env (torch 2.8 + fairchem-core 2.19). Output goes to netscratch.
#
# Usage: sbatch scripts/preprocess_omol25.slurm

#SBATCH -J omol25_prep
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/prep-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/prep-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_prep"
OUT_DIR=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/omol25_4m_processed
mkdir -p "$OUT_DIR"

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/omol25"

export PYTHONUNBUFFERED=1

echo "host: $(hostname)"
echo "start: $(date)"
echo "omol25 env: $(python -c 'import fairchem.core; print(fairchem.core.__version__)' 2>&1)"
echo ""

python -u scripts/preprocess_omol25.py \
    --src /n/netscratch/ryl_lab/Lab/omol25/train_4M \
    --out_dir "$OUT_DIR" \
    --max_atoms 200 \
    --val_frac 0.01

echo ""
echo "end: $(date)"
echo "disk usage of output:"
du -sh "$OUT_DIR"
ls -lh "$OUT_DIR"
