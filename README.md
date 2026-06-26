# Boltzmann-Guided Flow Matching (BGFM)

> **Read this if you do ML, not chemistry.** This README is written for a
> machine-learning audience. Chemistry is treated as a black box: wherever a
> domain term is unavoidable, it is followed by a bracketed gloss in ML terms,
> e.g. *Boltzmann distribution [the target is an energy-based model — an
> unnormalized density `p(x) ∝ exp(−E(x)/T)`]*. For the chemistry-native
> framing and the original project notes, see [`CLAUDE.md`](CLAUDE.md),
> [`AGENTS.md`](AGENTS.md), and the previous README preserved as
> [`README_backup.md`](README_backup.md).

**BGFM is a generative model over 3D point clouds with discrete per-point
labels, trained so that its learned density matches a *given* energy-based
model** — a pretrained "energy oracle" that scores any configuration. In the
application domain the point clouds are molecules (each point = an atom: a 3D
coordinate + a categorical type), and the energy oracle is a frozen universal
neural-network potential. The result is a single generator that not only
*imitates* the training set but is pushed to be *correctly weighted by energy*
(low-energy = high-probability), and it does so across a very wide input
distribution from one trained checkpoint.

Target venue: **ICLR 2027** (submission ~2026-09-25).

---

## TL;DR — the chemistry→ML dictionary

| Chemistry term | What it means for you (ML) |
|---|---|
| **Molecule** | a variable-size set of `N` points in `ℝ³`, each with a categorical label (1 of 83 classes) + an integer charge. A sample `x`. |
| **Boltzmann distribution** `p(x) ∝ exp(−E(x)/kT)` | an **energy-based model (EBM)** / unnormalized target density. `E` = energy (a cost), `kT` = temperature (a fixed scalar). Low energy ⇒ high probability. |
| **Energy** `E(x)` | a scalar **cost** assigned to a configuration by a frozen oracle. |
| **Force** `F(x) = −∇ₓE(x)` | the **negative gradient of the cost** w.r.t. coordinates — a per-coordinate target, the thing a *score* should match. |
| **(Universal) neural potential / OMol25** | a **frozen pretrained network** mapping `x ↦ (E, ∇E)`. Think *learned reward/cost model* or *differentiable simulator surrogate*. Covers 83 input classes (elements), including the hard ones (transition metals). |
| **Partition function** `Z` | the **intractable normalizing constant** of the EBM (`p = e^{−E/kT}/Z`). Per-sample-family and unknown. |
| **Score** | `∇ₓ log p(x)` — exactly what score matching estimates. |
| **DFT** | the **expensive ground-truth simulator** used (offline) to label the training data and to evaluate. |
| **SE(3)-equivariant** | the network respects 3D rotation+translation symmetry of the inputs. |
| **Conformer** | one low-energy 3D shape of a fixed graph — *a mode of a conditional distribution*. |
| **Bonds** | discrete graph edges over the points. **We do not model them** (see below). |

---

## The problem, in one paragraph

A standard 3D generative model (diffusion / flow matching) is trained by
**maximum-likelihood-style imitation** of a dataset. It learns to produce
samples that *look like* the data, and it is evaluated on dataset-imitation
proxies (validity / uniqueness / novelty — see [Related Work](#related-work)).
It never uses the fact that, in physics, there is a *correct* target density:
the **Boltzmann distribution** `p(x) ∝ exp(−E(x)/kT)` — an energy-based model
whose energy `E` is a real, learnable-from-physics quantity. BGFM closes that
gap: it trains a flow-matching generator **and** adds auxiliary losses that
force the generator's *own* density to agree with this EBM, using a frozen
energy oracle as the supervision signal.

Two things make this more than a footnote:

1. **The energy oracle is universal.** It is one pretrained network that scores
   *any* input over 83 classes (elements), not a hand-built cost for one fixed
   system. So a *single* BGFM checkpoint is energy-consistent across an
   enormous, heterogeneous input distribution — the regime where prior
   energy-targeting generators (Boltzmann generators) do not operate.
2. **It is cheap.** The naive way to match an EBM density needs the model's
   exact log-likelihood (an ODE + log-det term) and the intractable `Z`, and
   would backprop through the oracle. BGFM avoids all three (see
   [The tricks](#the-tricks-that-make-it-trainable)).

---

## Method

### The objective

```
L_total = L_FM  +  λ₁·L_force  +  λ₂·L_energy   (+ optional λ₃·L_anchor)
```

- **`L_FM` — flow matching (the imitation term).** Standard conditional
  flow matching: regress the velocity field on continuous coordinates
  (Gaussian prior, linear interpolant, velocity-MSE) and a discrete-flow
  cross-entropy on the categorical labels (element / charge). This alone is an
  ordinary 3D generator.

- **`L_force` — score ↔ energy-gradient matching (the *local* EBM term).**
  The score of an EBM is `∇ₓ log p = −∇ₓE / kT = F/kT`. So the model's
  *implied score* must equal `F/kT`. The trick: for the Gaussian flow-matching
  interpolant, the score is a **closed-form function of the velocity** the model
  already predicts:

  ```
  s(x_t, t) = (t·v_θ(x_t, t) − x_t) / ((1 − t)·σ²)
  ```

  No ODE, no `Z`, no backprop through the oracle — just read the score off the
  velocity at a few late times `t` (e.g. `[0.85, 0.92, 0.97]`) and regress it to
  the frozen target `F/kT`. This is the dense, every-step physics signal.
  *(The `(1−t)` denominator blows up at `t=1`; always probe `t<1`, and the
  per-point score norm is capped.)*

- **`L_energy` — log-density ↔ energy matching (the *global* EBM term).**
  The value form of Boltzmann: `log p(x) = −E(x)/kT − log Z`. Since `Z` (the
  normalizer) is unknown and family-specific, we don't hit the constant
  directly — instead we require `log p + E/kT` to *be* constant by minimizing
  its **variance within each sample family** (the `K` perturbations of one
  parent molecule). The within-family grouping cancels the per-family `log Z`.
  An optional tiny invariant **log-`Z` head** (`L_anchor`) pins the absolute
  level so the model can't cheat with a trivial constant.

The three terms coincide only in the infinite-data/infinite-capacity limit; in
practice each contributes a different gradient (dense velocity fit / local
gradient / global shape). A short appendix proves a **consistency** result: in
that idealized limit the global optimum of `L_total` is exactly the Boltzmann
distribution restricted to the data support. The paper's weight rests on the
*empirical* results, not the theory.

### The tricks that make it trainable

A naive EBM-density-matching term needs (a) the model's exact likelihood, (b)
the intractable `Z`, and (c) gradients through the oracle. Three moves remove
each:

1. **Frozen labels — no backprop through the oracle.** `E` and `F` are
   precomputed offline and stored as constant regression targets. The oracle
   network is never in the autograd graph.
2. **Score-from-velocity closed form — no ODE, no `Z`.** The `L_force` identity
   above turns score matching into a per-step algebraic read-off of the
   velocity. Runs every step, cheap.
3. **Variance form — cancels `Z`.** `L_energy` minimizes within-family variance
   of `log p + E/kT` instead of matching the unknown `−log Z`.

The only expensive residual is the true `log p` inside `L_energy` — a
continuous-normalizing-flow likelihood (**FFJORD**: reverse-time ODE +
divergence trace). It is made affordable with a **Hutchinson** divergence
estimator (one extra backward pass), a shallow ODE (~4–8 steps), a tiny batch,
and amortization (run it every `k`-th step). So `λ₂` is a light auxiliary nudge;
`L_force` carries the cheap dense signal. In the primary config `λ₂=0` (force
only); the energy term has its own config variants.

### Bond-free design

We generate points and labels but **not edges** (bonds). The oracle's training
data has no edge labels (the underlying physics has no discrete bond concept),
so supervising edges would inject wrong targets — especially on the hard
classes (metals). If a graph is needed downstream it is recovered post-hoc from
the generated geometry (`xyz2mol`). In config terms: `total_loss_weights.e = 0`.

### Architecture: monkey-patching, not forking

The backbone is **FlowMol3** — an SE(3)-equivariant, joint discrete+continuous
flow-matching generator (GVP-based message passing; continuous-time Markov
chain, "ctmc", for the discrete labels). We **do not fork it.** Instead we
patch a live `CTMCVectorField` instance at runtime:

- **Layer 1 — geometric hooks** (`cfm_mol/flow_model.py::patch_flowmol`): wraps
  5 methods on the vector field (interpolation, velocity, Euler step, discrete
  projections). Also holds inference-time score-guided sampling
  (`_apply_bgfm_score_guidance`).
- **Layer 2 — BGFM training loss**
  (`cfm_mol/bgfm_train_hook.py::patch_flowmol_bgfm`): wraps `training_step` to
  add `L_force` / `L_energy` / `L_anchor` on top of FlowMol3's `L_FM`.
- **Entry point** (`scripts/run_train.py`): reads the config, pops the `bgfm`
  sub-block (FlowMol3's `__init__` rejects unknown keys), builds the model,
  applies Layer 1 then Layer 2, and fits with PyTorch Lightning.

---

## Results (Jun 2026)

**Headline metric — per-sample-family perturbation R².** Take one held-out
molecule, perturb it into many nearby geometries (a small dataset of variants
of one sample), then correlate the model's `log p(x)` against the physical
`−E(x)/kT`. This directly measures *"does the generator rank configurations by
energy the way the EBM says it should?"* — `R²→1` means the model's density
ordering matches the energy ordering exactly; `slope→1` means it matches the
Boltzmann law quantitatively, not just in rank.

Best checkpoint (v7c, step 50k):

| Slice | R² | slope | frac `r > 0.5` |
|---|---|---|---|
| Overall (≤50 points/atoms) | **0.278** | 0.840 | 67.5% |
| Organic (the "easy", well-covered classes) | **0.380** | 0.800 | 65% |
| **Transition-metal (the hard, sparsely-covered classes)** | **0.501** | **1.40** | **80%** |

vs a **flow-matching-only baseline R² = 0.091** → **~3× tighter** alignment to
the EBM (slope 0.84 vs 0.29). The *strongest* slice is the hardest input
region (transition metals), which is the point: prior energy-targeting
generators don't operate there at all, so this is where the "universal" claim
earns its keep.

Planned (scripts ready): downstream "fewer optimization steps to reach a mode"
(relaxation savings), and few-step-ensemble-vs-long-simulation statistics
(sampling efficiency).

---

## Repo layout

```
bgfm/
├── cfm_mol/                       # our code
│   ├── bgfm_loss.py               # score-from-velocity, divergence (exact + Hutchinson), force/energy losses
│   ├── bgfm_density.py            # FFJORD log-density; per-family energy-variance + anchor losses
│   ├── bgfm_train_hook.py         # patch_flowmol_bgfm — the training-step wrapper (Layer 2)
│   ├── flow_model.py              # patch_flowmol — geometric hooks + inference score guidance (Layer 1)
│   ├── kt_conditioning.py         # optional temperature (kT) conditioning
│   ├── log_z_predictor.py         # tiny invariant log-Z head for the anchor term
│   ├── perturbation_loader.py     # iterator over precomputed K-perturbation shards (for L_energy)
│   ├── domain.py / fibre*.py      # geometric checks + projections (used by Layer-1 hooks)
│   ├── projection.py              # dead under bond-free; kept for a future paper
│   ├── physics*.py                # inference-time drift / cross-env physics client (reserved)
│   └── data/                      # dataset adapters: omol25.py (primary) + tmqm/kraken/radicals/hypervalent (eval)
├── configs/
│   ├── omol25_4m_bgfm.yaml        # PRIMARY training config (λ₁=0.1, λ₂=0, kT=1.0)
│   ├── omol25_4m_cfm.yaml         # flow-matching-only baseline (bgfm.enabled=false)
│   ├── geom_cfm_bondfree.yaml     # bond-free ablation on the backbone's home dataset
│   └── omol25_4m_bgfm_energy_v*.yaml   # energy-term experiment variants (room-T, T-conditional, energy-only)
├── scripts/
│   ├── run_train.py               # entry point (load cfg → patch → fit)
│   ├── preprocess_omol25.py       # raw data → backbone-native tensors (run in the omol25 env)
│   ├── precompute_energy_perturbations.py   # K perturbations/parent + frozen energy labels (for L_energy)
│   ├── eval_boltzmann_stage1.py   # compute log p on held-out variants (flowmol env)
│   ├── eval_boltzmann_stage2.py   # compute oracle energies + R² (omol25 env)
│   └── launch_*.sh / *.slurm      # cluster launchers (see Unity note below)
├── tests/                         # pytest: bgfm_loss, bgfm_density, loss weighting
├── CLAUDE.md                      # authoritative project guide (chemistry-native)
├── AGENTS.md                      # locked decisions + gotchas for AI agents
└── README_backup.md              # the previous (pre-pivot) README
```

---

## Environment (UMass Unity)

Two conda environments are required; a **torch version conflict forces the
split** — do not merge them.

| Env | Stack | Used for |
|---|---|---|
| `envs/flowmol` | torch 2.2 + DGL + PyTorch Lightning + FlowMol3 | **training & inference** |
| `envs/omol25`  | torch 2.8 + fairchem-core 2.19 (the oracle) | **data preprocessing & energy eval only** |

On Unity, conda is provided by the `miniforge3` module; the project envs live
under `envs/` (which is symlinked into scratch — see below — so they don't eat
the 100 GB `$HOME` quota):

```bash
module load miniforge3            # Unity's conda
# create once (envs land in envs/flowmol, envs/omol25 → scratch):
#   conda create -p ./envs/flowmol python=3.10   # then install torch 2.2 + DGL + flowmol3
#   conda create -p ./envs/omol25  python=3.11   # then install torch 2.8 + fairchem-core
conda activate ./envs/flowmol     # or ./envs/omol25
```

> **Unity note.** This repo was ported from a different cluster. The two
> project envs are **not yet built on this machine**, and the `*.slurm` / `*.sh`
> launchers and the `configs/*.yaml` paths still point at the original cluster's
> filesystem (`/n/holylabs/...`, `/n/netscratch/...`). On Unity, repoint each
> config's `output_dir` → `runs/` and `raw_data_dir` / `processed_data_dir` →
> `data/` & `processed_data/` (all symlinked to scratch), and adjust the SLURM
> `--partition` / `--account` to your Unity allocation
> (`pi_bsilva_umass_edu`). Do **not** install `fairchem-core` into the
> `flowmol` env (it pulls torch 2.8 + cudnn 9 and breaks DGL).

---

## Storage layout (UMass Unity)

`$HOME` is small (100 GB, 73% full) — **nothing big lives in the repo.** All
heavy artifacts (model weights, datasets, logs, conda envs) live in scratch and
are **symlinked back into the repo** so paths "just work":

```
~/bgfm/                                  # code only (this repo, in $HOME)
   ├── runs           ─┐
   ├── data            │   all symlinks →  ~/scratch_workspace/bgfm/<name>
   ├── processed_data  │   (~/scratch_workspace → /scratch4/workspace/...-hdp,
   ├── checkpoints     │    814 TB shared, purged periodically)
   ├── logs            │
   └── envs           ─┘
```

These symlink names are git-ignored (a symlink isn't matched by a trailing-slash
dir pattern, so they're also anchored explicitly in `.gitignore`), so the repo
stays code-only and the links never get committed. To reproduce the layout on a
fresh checkout:

```bash
SCRATCH=~/scratch_workspace/bgfm
mkdir -p "$SCRATCH"/{runs,data,processed_data,checkpoints,logs,envs}
cd ~/bgfm
for d in runs data processed_data checkpoints logs envs; do ln -sfn "$SCRATCH/$d" "$d"; done
```

Rule of thumb: **if it's a weight, a dataset, a log, or anything big, it goes
to `~/scratch_workspace/bgfm/` and is symlinked back — never committed.**

---

## Quick start

```bash
# 0. activate the training env
module load miniforge3 && conda activate ./envs/flowmol

# 1. unit tests (fast, no data needed)
python -m pytest tests/ -v
python -m pytest tests/test_bgfm_loss.py -v          # score, divergence, force/energy losses

# 2. preprocess data  (in the omol25 env — needs the oracle's stack)
conda activate ./envs/omol25
sbatch scripts/preprocess_omol25.slurm               # raw LMDB → backbone-native tensors
# for the energy term only: precompute K perturbations + frozen labels
sbatch scripts/run_precompute_energy.slurm

# 3. train  (in the flowmol env)
conda activate ./envs/flowmol
sbatch scripts/launch_omol25_bgfm_a100.sh            # BGFM (primary)
sbatch scripts/launch_omol25_level1_a100.sh          # flow-matching-only baseline

# 4. Boltzmann eval (two-stage, crosses envs)
conda activate ./envs/flowmol
python scripts/eval_boltzmann_stage1.py --checkpoint <ckpt> --config <cfg> ...   # → log p
conda activate ./envs/omol25
python scripts/eval_boltzmann_stage2.py --samples_json <stage1_out>/boltzmann_samples.json ...  # → energies + R²
```

Launchers accept `[config] [resume_ckpt]` positional args and support SIGUSR1
checkpoint-and-resume for preemptible (`gpu_requeue`-style) partitions. Always
launch with unbuffered output (`python -u`, `PYTHONUNBUFFERED=1`) so logs flush
under SLURM. (Adjust partitions/paths for Unity per the note above.)

---

## Config knobs (the `bgfm:` sub-block)

| Key | Values | Effect |
|---|---|---|
| `enabled` | true/false | master switch for the EBM-matching losses |
| `lambda_1` | float | weight on `L_force` (score↔energy-gradient) |
| `lambda_2` | float | weight on `L_energy` (log-density↔energy); `0` = off |
| `lambda_3` | float | weight on `L_anchor` (prevents trivial-constant solution) |
| `kT` | float | the EBM temperature scalar. `1.0` keeps `F/kT` numerically tame |
| `kT_conditioning` | true/false | sample `kT` per step and condition the model on it |
| `force_loss_type` | mse / cosine / norm_mse | full-score MSE vs direction-only variants |
| `force_target_mode` | true / shuffle_atoms | `shuffle_atoms` = negative-control ablation |
| `probe_mode` | path / endpoint | evaluate the score on the interpolant path or at the endpoint |
| `t_eval_values` | list[float] | late times to probe the score (e.g. `[0.85, 0.92, 0.97]`) |
| `force_correction_alpha` | float | shift the FM target by `α·F`; `0` = off |
| `warmup_frac` / `ramp_frac` | float | FM-only warmup, then linear ramp to full `λ` |
| `divergence_method` | hutchinson / exact | divergence estimator for the FFJORD likelihood |
| `energy_every_k_steps` | int | amortize the expensive `L_energy` (run every k-th step) |

Backbone-side knobs that matter: `total_loss_weights.{x,a,c,e}` (`e=0` ⇒
bond-free), `max_atoms` (=200), `dataset.atom_map` (the 83-class vocabulary),
`parameterization: ctmc`. Full reference in [`CLAUDE.md`](CLAUDE.md).

---

## Related Work

BGFM sits at the intersection of three otherwise-separate lines of work: (i) deep **3D molecular generators**, which learn to sample atom clouds but only *imitate a dataset*; (ii) **Boltzmann generators**, which do target the physical energy density p ∝ exp(−E/kT) [an energy-based model: density proportional to exp(−energy/temperature)] but classically for one fixed molecular system at a time; and (iii) **universal neural potentials**, fast learned surrogates for an expensive physics simulator [DFT — the ground-truth energy/force oracle] that *score* energies and forces for arbitrary chemistry but generate nothing. BGFM fuses all three: it turns a single universal potential into the *training target* of an unconditional 3D generator, so the generator's learned density is pushed to equal the Boltzmann distribution across the whole periodic table, in one trained model.

| Method | What it generates (method) | Domain / coverage | Eval metric | How BGFM differs |
|---|---|---|---|---|
| **EDM** — [Hoogeboom et al. (2022)](https://arxiv.org/abs/2203.17003) | 3D atoms (position+element+charge), bonds post-hoc; E(3) denoising diffusion | QM9 / GEOM organics (~5–16 main-group elements) | validity, atom/molecule stability, NLL | Dataset imitation, no energy target; BGFM adds force+energy losses tying density to exp(−E/kT) |
| **MiDi** — [Vignac et al. (2023)](https://arxiv.org/abs/2302.09048) | atoms+charges+**bonds**+3D jointly; mixed discrete/continuous diffusion | QM9 / GEOM organics | molecule stability, validity, bond metrics | Supervises bonds [human-abstraction edge labels]; BGFM is bond-free and Boltzmann-targeted |
| **EQGAT-diff** — [Le et al. (2024)](https://arxiv.org/abs/2309.17296) | 3D atoms+bonds; tuned equivariant diffusion, organic pretrain→finetune | QM9 / GEOM / PubChem3D organics | stability, validity, novelty | Transfer is dataset→dataset, not a physical potential; no energy objective |
| **FlowMol3** — [Dunn & Koes (2025)](https://arxiv.org/abs/2508.12629) | atoms+types+bonds+charges; SE(3) flow matching (BGFM's backbone) | GEOM / QM9 organics | validity, bond-length/angle distribution match | Same backbone, but pure imitation + bonds; BGFM adds physics losses, drops bonds, 83 elements |
| **ADiT** — [Joshi et al. (2025)](https://arxiv.org/abs/2503.03965) | molecules **and** crystals [periodic lattices]; latent diffusion transformer | QM9/GEOM organics + MP20 crystals | molecule + crystal validity/stability | Unifies across *data types* by coverage, not via a physical potential; no Boltzmann target |
| **Symphony** — [Daigavane et al. (2024)](https://arxiv.org/abs/2311.16199) | bond-free 3D atoms+types; autoregressive spherical-harmonic placement | QM9 organics (CHONF) | validity, atom/molecule stability | Imitation, narrow chemistry, no energy/force signal |
| **Torsional Diffusion** — [Jing et al. (2022)](https://arxiv.org/abs/2206.01729) | conformer [one low-energy shape of a *fixed* 2D graph] over torsion angles; diffusion + post-hoc reweighting | GEOM organics, classical force field | coverage, AMR; reweight ESS | Graph-conditioned; Boltzmann only post-hoc via classical energy. BGFM unconditional + universal potential, baked in |
| **ET-Flow** — [Hassan et al. (2024)](https://arxiv.org/abs/2410.22388) | conformers given a 2D graph; equivariant flow matching | GEOM organics | coverage, AMR | Conditional + dataset imitation; BGFM generates identity de novo with a Boltzmann objective |
| **Boltzmann Generators** — [Noé et al. (2019)](https://www.science.org/doi/10.1126/science.aaw1147) | equilibrium configs of one *fixed* molecule; normalizing flow + reweighting | single systems, classical energy | free-energy, reweighted observables | One model per system; samples conformations, not molecular identity; no cross-element transfer |
| **Equivariant Flow Matching** — [Klein et al. (2023)](https://arxiv.org/abs/2306.15030) | equilibrium configs of fixed system; equivariant CNF + OT flow matching | LJ/DW clusters, alanine dipeptide | ESS, free energy, NLL | Per-system fixed energy; BGFM uses one universal potential + generates element/charge |
| **Transferable Boltzmann Gen.** — [Klein & Noé (2024)](https://arxiv.org/abs/2406.14426) | conformations conditioned on a molecular graph; FM flow + reweighting | dipeptides (zero-shot to unseen) | zero-shot ESS, Boltzmann stats | Transfer only within peptide chemistry, conditional; BGFM is universal + unconditional |
| **iDEM** — [Akhound-Sadegh et al. (2024)](https://arxiv.org/abs/2402.06121) | samples for a fixed energy; diffusion sampler from energy/force only | model n-body energies (GMM, LJ13/55) | W2 distance, ESS | Per-system analytic energy, no chemical identity; BGFM is a multi-element generator |
| **DiG** — [Zheng et al. (2024)](https://www.nature.com/articles/s42256-024-00837-3) | structures conditioned on system descriptor; score diffusion + energy pretrain (PIDP) | proteins/ligands/catalysts, per-system FF/DFT | coverage vs MD, free energy | Conditional + per-domain energies; BGFM one universal potential, unconditional |
| **Adjoint Sampling** — [Havens et al. (2025)](https://arxiv.org/abs/2504.11713) | conformers given a 2D graph; on-policy energy-only diffusion sampler on an NN potential | drug-like organics, eSEN (10 main-group elements) | coverage/recall/precision, W2, ESS | Graph-conditioned, ~10 elements, on-policy; BGFM unconditional, 83 elements, off-policy FM-derived score |
| **OMol25** — [Levine et al. (2025)](https://arxiv.org/abs/2505.08762) | *not a generator* — DFT-surrogate energy/force oracle | 83 elements incl. transition metals, ≤~350 atoms | energy/force MAE vs DFT | The oracle BGFM *consumes*; it scores geometries but cannot sample |
| **UMA** — [Wood et al. (2025)](https://arxiv.org/abs/2506.23971) | *not a generator* — universal MLIP (energy/force/stress) | molecules + materials + catalysis, 83+ elements | energy/force/stress MAE | Predictor, not sampler; supplies E/F, BGFM is the generative layer on top |
| **Flow Matching** — [Lipman et al. (2023)](https://arxiv.org/abs/2210.02747) | generic samples; simulation-free CNF velocity regression | domain-agnostic (images) | NLL/bits-per-dim, FID | Pure data imitation, no chemistry/energy; BGFM's base objective, extended with physics losses |

#### 3D molecular generators

The dominant line learns to sample 3D atoms (and often bonds) de novo with SE(3)-equivariant diffusion or flow matching: EDM ([Hoogeboom et al., 2022](https://arxiv.org/abs/2203.17003)), its latent variant GeoLDM ([Xu et al., 2023](https://arxiv.org/abs/2305.01140)), joint 2D+3D models MiDi ([Vignac et al., 2023](https://arxiv.org/abs/2302.09048)), JODO ([Huang et al., 2023](https://arxiv.org/abs/2305.12347)) and MUDiff ([Hua et al., 2024](https://arxiv.org/abs/2304.14621)), the design-space study EQGAT-diff ([Le et al., 2024](https://arxiv.org/abs/2309.17296)) and geometry-complete GCDM ([Morehead & Cheng, 2024](https://arxiv.org/abs/2302.04313)); the flow-matching family FlowMol ([Dunn & Koes, 2024](https://arxiv.org/abs/2404.19739)) / FlowMol3 ([Dunn & Koes, 2025](https://arxiv.org/abs/2508.12629)) — BGFM's backbone — EquiFM ([Song et al., 2023](https://arxiv.org/abs/2312.07168)), SemlaFlow ([Irwin et al., 2025](https://arxiv.org/abs/2406.07266)) and Megalodon ([Reidenbach et al., 2025](https://arxiv.org/abs/2505.18392)); the autoregressive lineage G-SchNet ([Gebauer et al., 2019](https://arxiv.org/abs/1906.00957)), cG-SchNet ([Gebauer et al., 2022](https://www.nature.com/articles/s41467-022-28526-y)) and Symphony ([Daigavane et al., 2024](https://arxiv.org/abs/2311.16199)); the graph-only baseline DiGress ([Vignac et al., 2023](https://arxiv.org/abs/2209.14734)); and the cross-domain ADiT ([Joshi et al., 2025](https://arxiv.org/abs/2503.03965)). A related conditional sub-line generates conformers [one 3D shape of a *known* 2D graph — like one mode of a conditional distribution] given fixed connectivity: GeoMol ([Ganea et al., 2021](https://arxiv.org/abs/2106.07802)), GeoDiff ([Xu et al., 2022](https://arxiv.org/abs/2203.02923)), MCF ([Wang et al., 2023](https://arxiv.org/abs/2311.17932)), ET-Flow ([Hassan et al., 2024](https://arxiv.org/abs/2410.22388)), SO(3)-averaged flow ([Cao et al., 2025](https://arxiv.org/abs/2507.09785)) and FlexiFlow ([2025](https://arxiv.org/abs/2511.17249)). All of these optimize *dataset-imitation* metrics — validity, atom/molecule stability, uniqueness, novelty, and conformer coverage/AMR — and none enforce an energy/Boltzmann target; where energy appears (SemlaFlow, GCDM, Megalodon) it is only a post-hoc quality *diagnostic*, never the training objective, and coverage is restricted to main-group organic chemistry.

#### Boltzmann generators & energy-guided sampling

A second line does target the physical Boltzmann density p ∝ exp(−E/kT), classically by training one normalizing flow per *fixed* system against that system's energy: Boltzmann Generators ([Noé et al., 2019](https://www.science.org/doi/10.1126/science.aaw1147)), equivariant flows ([Köhler et al., 2020](https://proceedings.mlr.press/v119/kohler20a.html); [Klein et al., 2023](https://arxiv.org/abs/2306.15030)), energy-only samplers FAB ([Midgley et al., 2023](https://arxiv.org/abs/2208.01893)) and iDEM ([Akhound-Sadegh et al., 2024](https://arxiv.org/abs/2402.06121)), and the score-equals-force denoising force field ([Arts et al., 2023](https://arxiv.org/abs/2302.00600)). Recent work broadens transferability — across peptides for Timewarp ([Klein et al., 2023](https://openreview.net/forum?id=EjMLpTgvKH)), Transferable Boltzmann Generators ([Klein & Noé, 2024](https://arxiv.org/abs/2406.14426)), Sequential Boltzmann Generators ([Tan et al., 2025](https://arxiv.org/abs/2502.18462)) and force-guided FBM ([Yu et al., 2025](https://arxiv.org/abs/2408.15126)); across system *size* for materials ([Schebek et al., 2025](https://arxiv.org/abs/2509.25486)); and via the amortized, NN-potential samplers Adjoint Sampling ([Havens et al., 2025](https://arxiv.org/abs/2504.11713)) and its Schrödinger-bridge follow-up ([Liu et al., 2025](https://arxiv.org/abs/2506.22565)), plus GFlowNet torsion sampling ([Volokhova et al., 2024](https://arxiv.org/abs/2310.14782)). DiG ([Zheng et al., 2024](https://www.nature.com/articles/s42256-024-00837-3)) is the closest in spirit — an equilibrium-targeted generative model — and Energy Matching ([Balcerak et al., 2025](https://arxiv.org/abs/2504.10612)) unifies FM with EBMs. But these remain narrow relative to universal chemistry: they use a classical force field or per-system DFT (not one universal 83-element potential), most are *conditional* samplers of a known molecule's conformations rather than unconditional generators of element identity and charge, and energy-guided 3D generators like EEGSDE ([Bao et al., 2023](https://arxiv.org/abs/2209.15408)) steer toward learned *property* networks, not a physical Boltzmann density.

#### Universal neural potentials

The third line provides the *signal source* BGFM consumes — fast learned surrogates for expensive physics [DFT] that predict energy and forces [the negative gradient of energy w.r.t. coordinates — a per-coordinate training target like a score] for arbitrary geometry: OMol25 ([Levine et al., 2025](https://arxiv.org/abs/2505.08762)), the universal model family UMA ([Wood et al., 2025](https://arxiv.org/abs/2506.23971)) and its smooth, energy-conserving backbone eSEN ([Fu et al., 2025](https://arxiv.org/abs/2502.12147)), the materials foundation potential MACE-MP-0 ([Batatia et al., 2023](https://arxiv.org/abs/2401.00096)), the organic-only MACE-OFF ([Kovács et al., 2023](https://arxiv.org/abs/2312.15211)), the cross-domain pretrained backbone JMP ([Shoghi et al., 2023](https://arxiv.org/abs/2310.16802)), and the fast Orb potential ([Neumann et al., 2024](https://arxiv.org/abs/2410.22570)). These are *predictors*, not generators — they score a configuration but cannot sample new molecules — and BGFM uses one of them (OMol25, the molecular-accuracy oracle spanning 83 elements including transition metals) to define its target density. The closed-form density machinery BGFM reuses to evaluate log p — instantaneous change-of-variables with a Hutchinson trace estimator (FFJORD, [Grathwohl et al., 2019](https://arxiv.org/abs/1810.01367)) and the score/SDE view ([Song et al., 2021](https://arxiv.org/abs/2011.13456); DDPM, [Ho et al., 2020](https://arxiv.org/abs/2006.11239); OT-CFM, [Tong et al., 2023](https://arxiv.org/abs/2302.00482)) — comes from the generative-modeling toolbox, not from chemistry.

**BGFM's gap.** No prior method occupies the single cell BGFM fills: an *energy/Boltzmann-targeted density* (not dataset imitation) defined by a *universal cross-element potential* (organic + organometallic + transition-metal chemistry, not one fixed system or force field), realized as *one unconditional model trained once* that generates atom positions, element types, and charges de novo. Existing 3D generators imitate data and ignore energy; Boltzmann generators match energy but per fixed system and usually conditioned on a known molecule; universal potentials supply energy but generate nothing. BGFM bridges them by injecting a universal potential's forces and energies as auxiliary consistency losses on a flow-matching generator, and verifies the result directly with per-molecule perturbation R² (log p vs −E/kT) — a Boltzmann-calibration metric none of the imitation-trained generators can report.

---

## Locked design decisions (do not re-litigate)

See [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) for rationale and dates:

1. **Flow matching, not a reflected SDE** — `O(Δt)` vs `O(√Δt)` convergence, cleaner theory.
2. **Bond-free** (`total_loss_weights.e = 0`) — no edge supervision; edges are post-hoc.
3. **Three-term loss** `L_FM + λ₁L_force + λ₂L_energy` with frozen labels and the within-family variance form.
4. **`max_atoms = 200`** (large enough to cover the downstream catalyst complexes).
5. **The universal-oracle dataset (OMol25) is primary**, not the small-organics benchmarks (QM9/GEOM).

## Honest limitations

- **EBM-consistent w.r.t. the oracle, not ground-truth physics** — accuracy is
  bounded by the frozen potential.
- **Single temperature** by default (a `kT`-conditioned variant exists but isn't the headline).
- **`L_energy` uses an offline perturbation subset** — generalization from it is measured and is the main attackable point.
- **No edges** — connectivity is post-hoc.
- **Memory ceiling** — the FFJORD backward pass caps batch size.

## License

Our code: MIT. Any vendored baselines retain their original licenses.
