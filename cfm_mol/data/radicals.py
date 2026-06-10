"""Open-shell radical OOD slice (ICLR E1 slice #4).

Target: carbon-centred and nitrogen-centred radicals (TEMPO, DPPH, nitroxide
variants, alkyl + aryl C-radicals). Characterised by: total unpaired spin
> 0, atoms with odd valence, species not in QM9/GEOM training distribution.

Two sources considered:
  1. GEOM-Drugs radical subset: filter molecules where RDKit reports
     num_radical_electrons > 0.
  2. Hand-curated TEMPO / DPPH / nitroxide library from PubChem.

We use (1) for scale (~5k radicals in GEOM) and (2) for quality controls.

Baselines' failure mode here: QM9/GEOM soft-loss models have near-zero mass
on odd-valence configurations; they either skip the radical atom or add a
spurious H to "fix" valence.
"""
from __future__ import annotations

from pathlib import Path


SLICE_NAME = "radicals"
ATOM_MAP = ["C", "H", "N", "O", "F"]   # GEOM-Drugs atoms suffice for most radicals


def filter_geom_radicals(geom_processed_dir: Path, dest_dir: Path) -> None:
    """Filter GEOM-Drugs for molecules with unpaired electrons.

    Implementation sketch:
      1. Load GEOM processed data.
      2. For each molecule, reconstruct RDKit mol; run `Chem.GetFormalCharge`
         + `atom.GetNumRadicalElectrons()`. Keep molecules where any atom
         has radical electrons > 0 OR the molecule has odd # electrons.
      3. Write filtered subset to dest_dir in FlowMol3 processed format.
    """
    raise NotImplementedError(
        "Radical filtering: planned Week 7 of ICLR timeline. Needs GEOM "
        "processed data + rdkit in env."
    )


def curate_nitroxides(dest_dir: Path) -> None:
    """Hand-curated TEMPO / DPPH / nitroxide library (~30 molecules) for
    targeted quality control on specific radical motifs."""
    raise NotImplementedError("Planned Week 7.")
