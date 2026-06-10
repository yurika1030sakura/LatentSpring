#!/bin/bash
#SBATCH -J omol25_4m_dl
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/dl-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/dl-%j.err

set -e

DATA_ROOT=/n/netscratch/ryl_lab/Lab/omol25
mkdir -p "$DATA_ROOT"
mkdir -p /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl
cd "$DATA_ROOT"

echo "host: $(hostname)"
echo "start: $(date)"
echo "disk free: $(df -h . | tail -1)"
echo ""

# The OMol25 4M training subset (~19 GB compressed). CC-BY-4.0 license.
# Source: https://huggingface.co/datasets/facebook/OMol25
# CDN fallback (might be available): dl.fbaipublicfiles.com
echo "=== downloading train_4M.tar.gz ==="
if [ ! -f train_4M.tar.gz ]; then
    wget -c "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/train_4M.tar.gz" \
        -O train_4M.tar.gz 2>&1 | tail -5 || echo "[WARN] CDN train_4M download failed; trying HF"
fi

# val_neutral is ~119MB - small but useful for quick eval
echo "=== downloading val_neutral.tar.gz ==="
if [ ! -f val_neutral.tar.gz ]; then
    wget -c "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val_neutral.tar.gz" \
        -O val_neutral.tar.gz 2>&1 | tail -5 || echo "[WARN] val_neutral download failed"
fi

ls -lh *.tar.gz
echo ""
echo "=== extracting train_4M ==="
if [ -f train_4M.tar.gz ] && [ ! -d train_4M ]; then
    tar -xzf train_4M.tar.gz
fi

echo "=== extracting val_neutral ==="
if [ -f val_neutral.tar.gz ] && [ ! -d val_neutral ]; then
    tar -xzf val_neutral.tar.gz
fi

echo ""
echo "=== final disk layout ==="
du -sh train_4M/ val_neutral/ *.tar.gz 2>&1 | head -10

echo ""
echo "done: $(date)"
