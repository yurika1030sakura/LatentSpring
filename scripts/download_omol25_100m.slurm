#!/bin/bash
#SBATCH -J omol25_100m_dl
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/100m-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/100m-%j.err

set -e

DATA_ROOT=/n/netscratch/ryl_lab/Lab/omol25_100m
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "host: $(hostname)"
echo "start: $(date)"
echo "disk free: $(df -h . | tail -1)"
echo ""

# OMol25 full training set (456 GB compressed, ~1 TB uncompressed).
# Source: dl.fbaipublicfiles.com CDN
# CC-BY-4.0 license

echo "=== downloading train.tar.gz (456 GB) ==="
if [ ! -f train.tar.gz ] || [ $(stat -c %s train.tar.gz 2>/dev/null || echo 0) -lt 400000000000 ]; then
    wget -c "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train.tar.gz" \
        -O train.tar.gz
fi
ls -lh train.tar.gz

echo ""
echo "=== extracting train (this takes ~1-2 hours) ==="
if [ ! -d train ]; then
    tar -xzf train.tar.gz
fi
du -sh train/ 2>&1

echo ""
echo "done: $(date)"
