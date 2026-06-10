#!/bin/bash
# Download + process all small OOD slices for ICLR E1:
#   - tmQM (git clone, ~400 MB, keep Pd/Ni/Rh/Ir subset only)
#   - kraken (Zenodo tar.gz, ~50 MB, keep unusual-P subset)
#   - hypervalent (hand-curated SMILES + RDKit 3D embedding, tiny)
#
# Each target emits its own val_data_processed.pt under a separate
# processed_dir so they can be evaluated independently. All three run
# sequentially inside one SLURM job (combined < 30 min even on a single
# node); the large slice (radicals from GEOM) waits on the geom SLURM
# and is kicked off separately.
#
# Radicals (slice #4) is skipped here because it needs processed GEOM
# as input -- see download_geom.slurm, then run scripts/process_radicals.py
# as a follow-up.

#SBATCH -J cfm_ood
#SBATCH -p sapphire,shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 3:00:00
#SBATCH -o /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/ood-%j.out
#SBATCH -e /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/runs/ood-%j.err

set -e

PROJ=/n/holylabs/ryl_lab/Lab/yulili_cfm_mol

echo "host: $(hostname)"
echo "start: $(date)"

source /n/sw/Mambaforge-23.3.1-1/etc/profile.d/conda.sh
conda activate "$PROJ/envs/flowmol"

cd "$PROJ"

DATA_ROOT=$PROJ/data
mkdir -p "$DATA_ROOT"

# ---------------------------------------------------------------------------
# tmQM
# ---------------------------------------------------------------------------
echo ""
echo "=== tmQM (broad: 15 metals incl. Cu/Fe/Zn) ==="
python -m cfm_mol.data.tmqm \
    --raw_dir "$DATA_ROOT/tmqm" \
    --processed_dir "$DATA_ROOT/tmqm_broad_processed" \
    --mode broad \
    --download || echo "[WARN] tmQM broad failed"

echo ""
echo "=== tmQM (narrow: Pd/Ni/Rh/Ir catalysis sub-slice) ==="
python -m cfm_mol.data.tmqm \
    --raw_dir "$DATA_ROOT/tmqm" \
    --processed_dir "$DATA_ROOT/tmqm_narrow_processed" \
    --mode narrow || echo "[WARN] tmQM narrow failed"

echo ""
echo "=== tmQM (broad, metal-stripped: 10-organic atom_map, ligand framework only) ==="
python -m cfm_mol.data.tmqm \
    --raw_dir "$DATA_ROOT/tmqm" \
    --processed_dir "$DATA_ROOT/tmqm_organic_processed" \
    --mode broad --strip_metal || echo "[WARN] tmQM organic failed"
echo "tmqm done: $(date)"

# ---------------------------------------------------------------------------
# kraken
# ---------------------------------------------------------------------------
echo ""
echo "=== kraken ==="
python -m cfm_mol.data.kraken \
    --raw_dir "$DATA_ROOT/kraken" \
    --processed_dir "$DATA_ROOT/kraken_processed" \
    --download --unusual_P_only || echo "[WARN] kraken failed"
echo "kraken done: $(date)"

# ---------------------------------------------------------------------------
# hypervalent (no download; SMILES embedded in the module)
# ---------------------------------------------------------------------------
echo ""
echo "=== hypervalent ==="
python -m cfm_mol.data.hypervalent \
    --raw_dir "$DATA_ROOT/hypervalent" \
    --processed_dir "$DATA_ROOT/hypervalent_processed" \
    || echo "[WARN] hypervalent failed"
echo "hypervalent done: $(date)"

echo ""
echo "=== all OOD slices done: $(date) ==="
ls -lh "$DATA_ROOT"/*_processed/*.pt 2>/dev/null | head
