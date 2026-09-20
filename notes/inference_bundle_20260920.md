# LatentSpring EGNN inference bundle

`cfm_mol.latentspring_generator.LatentSpringGenerator` loads inference weights
without importing the research runners or the energy-oracle interface. A bundle
contains frozen parent, physical-head, and hydrogen-flow states. It contains no
optimizer states, training examples, or energy model. Hashes verify every weight
file and the loaded tensor states.

```python
from cfm_mol.latentspring_generator import LatentSpringGenerator

generator = LatentSpringGenerator("bundle_directory", device="cuda")
samples = generator.generate(
    [6, 6, 8, 1, 1, 1, 1, 1, 1],
    charge=0, spin_multiplicity=1, samples=16, seed=42,
    fit=0, family="fm",
)
coordinates = samples["positions"]  # Angstrom; every requested output
```

The fixed recipe applies the strength-four physical head, followed by the
optional four-step hydrogen flow. `hydrogen=False` returns the physical-head
output directly. The result also retains initial coordinates, the coordinates
before the H readout, the activation mask, and measured call counts. Neither
mode evaluates an energy model or filters outputs by validity or energy.

The trained checkpoints cover neutral organic singlets. Inputs with incompatible
charge/spin or electron parity are rejected. The implementation accepts up to200
atoms, but the confirmed shared-EGNN readout studies cover17--40 atoms; the small
example above demonstrates the input format rather than a performance claim.

The upstream EDM/EGNN implementation is required at the exact source hashes in
the manifest. On a different machine, pass `upstream="/path/to/edm"`; weights
and source hashes are still checked. `stream` selects a nonnegative condition
stream within the seed, matching the experimental generation convention. Batch
size8 is fixed and recorded because it affects the diffusion RNG sequence.

```bash
python -s -m scripts.generate_from_bundle \
  --bundle bundle_directory --condition condition.json --device cuda \
  --family fm --fit 0 --samples 16 --seed 42 --out new_samples
```

The exporter is `scripts.research.export_inference_bundle`. It creates compact
copies from completed fit records and preserves the original checkpoint hashes.
The two-fit development package is in
`runs/seed_replication_v1/bundle_prototype`; a five-fit package can be exported
only after every corresponding study has completed. These local weight packages
are not stored in Git; source code and reproducibility receipts are tracked.
