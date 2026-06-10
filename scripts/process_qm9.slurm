#!/bin/bash
# Process QM9 only (env already built by install_and_process.slurm which
# died at QM9 step with shared-memory exhaustion). Use more RAM + fewer
# workers to avoid the torch multiprocessing mmap crash we hit at 12%
# in the previous run.

#SBATCH -J qm9_proc
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 6:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/qm9proc-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/qm9proc-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol
FLOWMOL=$PROJ/baselines/flowmol3
cd "$FLOWMOL"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

echo "host: $(hostname)"
echo "start: $(date)"
echo "mem: $(free -h | head -2 | tail -1)"
echo ""

# Go SERIAL: torch multiprocessing allocates tensors via /dev/shm which
# is capped at ~50% of RAM regardless of --mem request. Past run crashed
# at exactly 12% (=11891 mols) every time from /dev/shm exhaustion.
# Serial is ~4 min total at 400 mol/sec, slower-than-multiproc but reliable.
python process_qm9.py \
    --config "$PROJ/configs/qm9_kekulized.yaml" \
    --n_cpus 1

echo ""
echo "done: $(date)"
ls -lh "$PROJ/data/qm9_processed/"
