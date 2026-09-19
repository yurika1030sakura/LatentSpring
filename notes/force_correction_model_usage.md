# Running the tested force-correction variant

Use the existing flowmol environment. The recipe and both checkpoint hashes are
in `configs/research/latentspring_force_correction_v1.json`. The full checkpoints
remain in HOLY storage, not in the GitHub source archive.

From the HOLY project checkout:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/flowmol/bin/python -u \
  -m scripts.generate_latentspring \
  --checkpoint runs/context_confirmation_v1/s0/sampling/pair_long.ckpt \
  --config configs/sweep/a1_fm_only_s2.yaml \
  --conditions /path/to/conditions.json \
  --out /path/to/new_output_folder \
  --samples 32 --midpoint-steps 32 --seed 57051 --device cuda \
  --allow-legacy-pickle
```

A condition file has `rows`, each containing `atomic_numbers`, `charge`, and
`spin_multiplicity`. Coordinates and supplied bond graphs are not inputs. The
output NPZ files retain every raw generated structure and its starting coordinates;
`generation.json` records hashes and actual network-call counts. No energy model,
ranking, rejection or coordinate optimization is part of generation.

The correction contains8178 learned parameters on top of the frozen5.90M parent.
It uses128 backbone and64 head calls per output. The shared parent is the same
in both checkpoint variants; their head-fitting seeds differ. Current evidence
covers the stated neutral-organic panels, not every possible element, charge,
size or temperature. The model does not guarantee equilibrium isomer frequencies
or convergence to an energy minimum.
