#!/bin/bash
# Extract 456GB OMol25 100M tar.gz to aselmdb shard layout expected by
# AseDBDataset. Walltime sized for typical tar -xzf on NVMe scratch
# (~3-5 hours for 456GB compressed).

#SBATCH -J omol25_100m_extract
#SBATCH -p sapphire
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/extract-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/omol25_dl/extract-%j.err

set -e

DATA_ROOT=/n/netscratch/ryl_lab/Lab/omol25_100m
cd "$DATA_ROOT"

echo "host: $(hostname)"
echo "start: $(date)"
echo "disk free before:"
df -h . | tail -1
echo ""
echo "tar.gz size: $(du -h train.tar.gz)"
echo ""
echo "=== extracting train.tar.gz ==="
if [ ! -d train ]; then
    tar -xzf train.tar.gz
fi
echo ""
echo "disk free after:"
df -h . | tail -1
echo "train/ size: $(du -sh train 2>&1 | head -1)"
echo "number of shards: $(ls train/ | wc -l)"
echo ""
echo "done: $(date)"
