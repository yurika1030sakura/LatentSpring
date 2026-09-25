# LatentSpring

LatentSpring generates three-dimensional molecular coordinates from atomic composition. A random-tree harmonic source supplies the starting coordinates. A flow-matching model transports them toward molecular structures, and two small networks correct its predicted endpoint: one learns geometric recovery from perturbed reference structures, and the other learns a force-based displacement.

The input is an ordered list of atomic numbers. The released models use neutral organic singlets containing carbon and hydrogen. Sampling returns Cartesian coordinates in angstroms; atom identities remain fixed throughout the trajectory. Energy and force labels are used during training. Inference uses the saved neural networks.

## Installation

Use Python 3.10 and a separate environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-inference.txt
```

The inference environment contains PyTorch 2.2.0, NumPy 1.26.3, SciPy 1.12.0, and RDKit 2025.3.2. Both CPU and CUDA inference are supported. The required EDM source files and their MIT license are included under `vendor/edm` in the release archives.

The force-teacher experiments use a separate environment with PyTorch 2.8 and fairchem-core 2.19. Keep that environment separate from the inference and FlowMol environments.

## Code and weights

The submission archives contain the code, exact inference weights, example inputs, and saved evaluation arrays. Place the `weights` and `vendor` directories beside this README. The weights are ordinary PyTorch state dictionaries without optimizer states. `weights/manifest.json` records each file's checksum and the components used by each model.

Download the archives from the [model release](https://github.com/yurika1030sakura/LatentSpring/releases/tag/iclr2027-models).

Two packages are provided:

- `latentspring_supplement.zip`: implementation, result arrays, vendored dependencies, and all five final LatentSpring fits.
- `latentspring_supplement_full.zip`: the same files plus baseline, continuation-control, transfer, and optional hydrogen-readout weights.

The full package contains 43 weight files and 41 model configurations. Shared networks are stored once. The five main configurations are `latentspring_s0` through `latentspring_s4`.

## Generate structures

From the extracted archive directory:

```bash
python -m scripts.sample \
  --weights weights \
  --condition examples/condition.json \
  --model latentspring_s0 \
  --samples 16 \
  --seed 79151 \
  --device cpu \
  --out samples/example
```

Use `--device cuda` to run on a GPU. The example supplies the C7H15N composition shown in the method figure. A condition file has three fields:

```json
{"numbers": [6, 6, 8, 1, 1, 1, 1, 1, 1], "charge": 0, "spin_multiplicity": 1}
```

`numbers` lists every atom, including hydrogens. The implementation accepts at most 200 atoms and checks the available element vocabulary. The main evaluation covers compositions with 17--40 atoms; the backbone training set contains structures with 8--64 atoms.

Each run writes:

| File | Contents |
|---|---|
| `samples.xyz` | Every generated structure, in input atom order |
| `samples.pt` | Coordinates, initial source draws, conditions, and sampling settings |
| `report.json` | Model name, seed, sample count, and network-evaluation costs |

Every requested sample is returned. Bond inference and physical evaluation are separate from generation.

The Python interface is:

```python
from cfm_mol.release import Generator

model = Generator("weights", device="cuda")
result = model.generate(
    [6, 6, 8, 1, 1, 1, 1, 1, 1],
    model="latentspring_s0",
    samples=16,
    seed=123,
)
coordinates = result["positions"]
```

## Model configurations

| Name | Configuration | Fits |
|---|---|---:|
| `latentspring_sN` | Recovery-trained FM with geometric and physical corrections | 0--4 |
| `fm_sN` | Initial self-conditioned harmonic FM | 0--4 |
| `fm_physical_sN` | Initial FM with its physical correction | 0--4 |
| `gaga_sN` | GAGA baseline | 0--4 |
| `gaga_physical_sN` | GAGA with its own physical correction | 0--4 |
| `gaussian_fm_sN` | One-pass Gaussian-source FM | 0--1 |
| `harmonic_fm_sN` | One-pass harmonic-source FM | 0--1 |
| `edm_sN` | EDM baseline | 0--1 |
| `continuation_sN` | Unperturbed endpoint-continuation control with physical correction | 0--1 |
| `{gaussian_fm,harmonic_fm,edm,gaga}_transfer_sN` | Native parent with both FM-trained correction networks | 0--1 |

The optional `--hydrogen` flag is available for the initial `fm_physical_sN` and `gaga_physical_sN` configurations used in the hydrogen-readout study. The main LatentSpring comparison uses the coordinates before this readout.

A final-model sample uses 128 backbone evaluations, 64 geometric-head evaluations, and 64 physical-head evaluations. The geometric head has 7,747 parameters; the physical head has 7,106. Their strengths are one and four, respectively.

## Reproduce the comparisons

The numerical comparisons can be reproduced with NumPy alone:

```bash
python -m scripts.reproduce --results results --out reproduced_results.json
```

This command reads the saved per-output arrays, recomputes the main comparison, source/correction transfer results and common-element panel, and checks the source effects, matched-continuation comparison, and the three additional fits' paired bootstrap intervals against the reported values.

On the 64-composition evaluation panel:

| Method | Fits | Graph validity | Graph + force | Geometry + force |
|---|---:|---:|---:|---:|
| Gaussian FM | 2 | 16.70% | 6.25% | 1.51% |
| EDM | 2 | 22.27% | 8.15% | 2.83% |
| GAGA | 5 | 22.60% | 8.75% | 2.40% |
| LatentSpring | 5 | 35.49% | 35.35% | 9.63% |

Graph validity requires a connected, sanitized inferred molecular graph. The force criterion is GFN2-xTB atomic force RMS at most 5 eV/Å. Geometry-and-force yield additionally applies the stated checks of bond lengths, angles, internal distances, planarity, and assigned radicals. Each rate includes all attempts. The physical scores are evaluated at the raw generated coordinates.

The paired-coordinate replay check is available in the full package:

```bash
python -m scripts.research.verify_submission_release \
  --release . --device cuda --out verification/local_replay.json
```

The exported files were checked by loading all 41 configurations and replaying 240 archived trajectories. The manifest retains tensor hashes so that the released networks can be matched to the evaluated states.

## Training and implementation

The main experiments use 20,000 OMol25 training structures. Atomic composition separates the training and validation groups. Each initial FM backbone receives 15,000 two-pass updates, followed by 4,000 recovery updates. The geometric head receives 10,000 updates on 3,783 screened training references. Physical heads are trained from force targets at provisional endpoints of the initial generators.

| Component | Implementation |
|---|---|
| Harmonic tree mixture | `cfm_mol/tree_mixture_prior.py`, `cfm_mol/tree_prior_controls.py` |
| Shared EGNN and native samplers | `cfm_mol/matched_egnn.py` |
| Geometry self-conditioning | `cfm_mol/connectivity_feedback.py` |
| Backbone recovery objective | `cfm_mol/geometry_recovery.py` |
| Geometric correction | `cfm_mol/geometry_recovery_field.py` |
| Physical correction | `cfm_mol/physical_connection.py` |
| Flow/diffusion endpoint conversion | `cfm_mol/matched_physical_connection.py` |
| Release loader and sampling API | `cfm_mol/release.py` |
| Chemical geometry checks | `cfm_mol/chemical_geometry_review.py` |
| Paired bootstrap intervals | `cfm_mol/replication_statistics.py` |

Training and evaluation entrypoints are in `scripts/research`. The saved protocols under `research/evidence` specify dataset selections, seeds, update counts, teacher settings, and checkpoint hashes. To retrain, obtain OMol25 under its data-access terms and set the data, upstream-code, and force-teacher locations for your installation. Inference and score reproduction use the files supplied in the release.

Earlier FlowMol experiments use a separate PyTorch 2.2/DGL/Lightning environment. The main shared-EGNN release does not require FlowMol, DGL, or the force-teacher model for inference.

## Paper and attribution

The manuscript is [LatentSpring: Harmonic Sources and Physical Corrections for Molecular Generation](paper/latentspring.pdf). Its appendices give the source-density derivation, training settings, geometric criteria, and additional experiments.

The EGNN and diffusion implementation builds on [E(3) Equivariant Diffusion for Molecule Generation](https://github.com/ehoogeboom/e3_diffusion_for_molecules). The vendored source retains its MIT license and copyright notices. OMol25 and the other external datasets and models retain their original licenses and attribution requirements.
