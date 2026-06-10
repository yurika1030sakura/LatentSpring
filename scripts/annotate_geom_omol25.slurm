#!/bin/bash
# Cell B annotation: run OMol25 NN on every GEOM-Drugs molecule to produce
# DFT-level forces + energies. Output drops into a NEW processed dir
# that is structurally identical to the OMol25 4M output (just trained
# on GEOM data with OMol25 labels).

#SBATCH -J geom_omol25_annot
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/geom_annot-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_prep/geom_annot-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
mkdir -p "$PROJ/runs/omol25_prep"
SRC=$PROJ/baselines/flowmol3/data/geom_5_kekulized
OUT=/n/netscratch/ryl_lab/Lab/yulili_cfm_mol/geom_bgfm_processed

cd "$PROJ"
source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/omol25"

export PYTHONUNBUFFERED=1

echo "host: $(hostname)"; nvidia-smi -L; echo ""
echo "src: $SRC"
echo "out: $OUT"
echo ""

python -u scripts/annotate_geom_with_omol25.py \
    --src_dir "$SRC" \
    --out_dir "$OUT" \
    --device cuda \
    --splits train val test
