# Benchmarking published 3D molecular generators on OMol25

Goal: train existing de-novo 3D molecular generators on **OMol25** (the QM-native,
83-element, bond-free dataset BGFM uses) with **I/O changes only — no architecture
changes** — and evaluate each on (a) its **original** metric and (b) our **new
Boltzmann-consistency metric** (per-molecule R² of model `log p` vs `−E/kT` under
the OMol25 neural potential).

> **Novelty note:** generation *on OMol25* is **not** unclaimed (Zatom-1,
> arXiv 2602.22251; Flowr.root, 2510.02578). The open angle is the
> **Boltzmann-consistency** framing — verify before relying on it.

## Shared bond-free I/O shim

Every OMol25 molecule = `N` atoms with a 3D position, an element index in `[0,82]`
(contiguous atomic number `Z = index + 1`), and a formal charge. No bonds. Each
baseline gets a thin adapter that converts our processed tensors into that
baseline's native batch format; the model, losses, and sampler are untouched.

## Tier-A baselines (bond-free, fair comparators)

| Baseline | Env | Status | Files added | Source edits (patch) |
|---|---|---|---|---|
| **EDM** | `envs/edm` (torch 2.2/cu121, numpy<2, rdkit, matplotlib) | ✅ smoke-trains (train+val+test NLL) | `gen_omol25_config.py`, `qm9/omol25_data.py` | `edm_io.patch` (`get_dataset_info` + `retrieve_dataloaders`) |
| **GeoLDM** | reuses `envs/edm` | ✅ smoke-trains (AE first-stage) | same two files | `geoldm_io.patch` (identical edits) |
| **Symphony** | `envs/symphony` (JAX 0.4.x + tf 2.13, pinned) | ⏳ env + adapter ready; smoke pending | `omol25_to_npz.py`, `symphony/data/datasets/omol25.py` | `symphony_io.patch` (`datasets/utils.py` registry) |

### EDM / GeoLDM (PyTorch)

```bash
# 1. generate dataset_info from processed tensors (any torch env)
python gen_omol25_config.py <processed_dir>          # writes configs/omol25_config.py
# 2. smoke-train (in envs/edm)
python main_qm9.py --dataset omol25 --datadir <processed_dir> --no_wandb \
    --break_train_epoch True --n_epochs 1 --batch_size 8 --num_workers 0 \
    --nf 64 --n_layers 3 --diffusion_steps 50 --n_stability_samples 0 --exp_name omol25_smoke
# GeoLDM: same command (omit --train_diffusion to smoke the autoencoder stage)
```
I/O fixes that were needed (all in the patch / adapter, **no architecture change**):
numpy pinned <2 (torch 2.2 compat); `matplotlib` added; EDM split key is `valid`
(maps to our `val` file); size prior made dense over 1..200 (EDM keys node-size
dist on exact sizes); `drop_last=True` (EDM's `kl_prior.squeeze()` breaks on size-1
batches). `charges` = nuclear `Z` = atom-type index + 1.

### Symphony (JAX)

```bash
# 1. convert torch tensors -> torch-free npz (in a torch env, e.g. flowmol)
python omol25_to_npz.py <processed_dir> <processed_dir>/omol25_symphony.npz
# 2. dataset is registered as `omol25` in symphony/data/datasets/utils.py
#    (config.dataset = "omol25", config.root_dir = <processed_dir>)
```
Env note: `tensorflow==2.13` caps `numpy<=1.24.3` while modern `jax` needs
`numpy>=2` — pin `numpy==1.24.3` + `jax 0.4.x`, and skip the optional git
model-backbone deps (mace/nequip/allegro-jax).

## Evaluation (both dimensions)

- **Original metrics** — each baseline's native validity/stability/uniqueness/novelty
  (or conformer COV/MAT). For the 83-element/transition-metal regime, prefer the
  geometry-based `scripts/evaluate_validity.py` (xyz2mol) over each repo's
  QM9/GEOM valence tables.
- **Boltzmann R² (new)** — `scripts/eval_boltzmann_stage1.py` (model `log p` on
  perturbations; swap in each model's own likelihood — diffusion ELBO for
  EDM/GeoLDM, autoregressive `log p` for Symphony) → `scripts/eval_boltzmann_stage2.py`
  (OMol25 oracle energies + correlation). Stage 2 is model-agnostic and reused as-is.

## Out of scope (per recon)
DiGress (2D-only, no 3D coords → no energy), Torsional Diffusion (density on
torsion space → no full-coordinate `log p`), pure energy-samplers (no training
dataset). SemlaFlow / MiDi / ETFlow are Tier B (need a distance-based bond pass).
