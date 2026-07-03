# Hierarchical Boltzmann Flow Matching (HBFM / BGFM) — Project Document

*Canonical, current-version project reference (2026-07). Supersedes the pre-pivot
framing in `README.md` and the flat-BGFM parts of `CLAUDE.md`. Companion files:
`notes/appendix_hbc.tex` (formal theory), `notes/benchmark_plan.md` (tables + numbers).*

---

## 0. One-paragraph summary

We are building a **3D molecular generator whose learned probability density provably
equals the Boltzmann distribution** of a *universal* neural potential (OMol25), across the
whole periodic table, with **no bond supervision**. The core idea is to factor generation
into two levels — **which molecule** (composition `c`) and **its 3D shape** (conformation
`r`) — and to show that the joint generator is Boltzmann **iff** each level matches its own
target, with the two levels linked by a **conditional free energy** `F(c)`. This gives (a)
a theoretical contribution that is specific to molecular generation and previously unclaimed,
and (b) a model that can report physical quantities (equilibrium populations, relative
free energies) that ordinary generators cannot. Target venue: an ICLR oral.

---

# Part I — Background (read this first)

Everything below is the minimum you need before the method and experiments make sense.

## 1. Molecules: composition vs. conformation

A molecule in 3D is fully described by two things:

- **Composition `c` (the chemical identity):** which atoms it contains (elements), how many,
  the total **charge**, and spin. Two molecules with the same atoms but different connectivity
  (constitutional isomers) or protonation state are different `c`. Think "it is caffeine."
- **Conformation `r` (the 3D geometry):** the actual Cartesian coordinates of every atom,
  `r ∈ ℝ^{3N}`. The *same* molecule flexes into many shapes ("conformers"). Think "caffeine,
  folded this particular way."

Physics is invariant to rigid motions: translating or rotating a molecule doesn't change its
energy. So `r` really lives in `ℝ^{3N}` **modulo SE(3)** (translation + rotation). In practice
we remove the centre of mass and work rotation-aware.

**Bonds are an abstraction.** Chemists draw bonds, but nature only has nuclei + electrons.
Bond orders are a human bookkeeping device that breaks down for transition metals, radicals,
and hypervalent species. We therefore work **bond-free**: the model generates atoms +
coordinates; bonds (if wanted) are inferred afterward from the geometry.

## 2. Energy, DFT, and forces

Every geometry `r` of a molecule `c` has a **potential energy** `E(c, r)` — the electronic
ground-state energy of that arrangement of nuclei. As a function of `r`, `E` is the
**potential energy surface (PES)**: valleys are stable conformers, the deepest valley is the
global minimum, hills are transition states.

- **DFT (Density Functional Theory)** is the standard quantum-chemistry method that computes
  `E(c, r)` (and its gradient) by approximately solving the electronic structure. It is
  accurate but **expensive** — seconds to hours per geometry — so you cannot call it inside a
  training loop millions of times.
- **Force.** The force on the atoms is the negative gradient of the energy:
  `F(c, r) = −∇_r E(c, r)`. Forces point *downhill* in energy (toward more stable geometries).
  DFT returns forces alongside energy. (ASE convention: `atoms.get_forces()` returns `F = −∇E`.)

## 3. Machine-learned interatomic potentials (MLIPs) and OMol25

Because DFT is too slow, the field trains neural networks to **predict `E` and `F` from
`(atoms, positions)`** at near-DFT accuracy but ~milliseconds per call. These are **MLIPs**
(also "universal potentials" / "foundation potentials").

**OMol25 ("Open Molecules 2025")** is two things we use:

1. **A dataset** — ~100M (we use a 4M subset) DFT-labeled molecular geometries covering **83
   elements**, including charged, radical, and transition-metal systems. It is **QM-native**:
   each record stores positions + atomic numbers + **DFT forces and energy**, and **no bond
   orders**. This is exactly the right training data for a bond-free, physics-grounded generator.
2. **A pretrained MLIP** — the **eSEN** model (an energy-conserving equivariant network,
   FAIRChem/UMA family). Given `(atomic_numbers, positions, charge, spin)` it returns
   `E` (eV) and `F` (eV/Å). This is our **teacher / oracle**: a fast, differentiable stand-in
   for DFT that we can query during training and evaluation.

Throughout, "**the teacher**" = the OMol25 eSEN potential.

## 4. The Boltzmann distribution (the actual target)

At thermal equilibrium and temperature `T`, a physical system does not sit at the single
lowest-energy geometry — it fluctuates, visiting geometry `r` with probability given by the
**Boltzmann distribution**:

```
π(r | c) = exp(−E(c,r) / kT) / Z(c),        Z(c) = ∫ exp(−E(c,r)/kT) dr
```

- `kT` is temperature × Boltzmann's constant, an energy scale (room temperature ≈ 0.025 eV).
  We often train at `kT = 1 eV` for numerical stability and recover physical temperature via
  temperature conditioning; `kT` sets how sharply probability concentrates at low energy.
- `Z(c)` is the **partition function** (a.k.a. configurational integral): the normalizing sum
  over *all* geometries. It is generically **intractable** (a high-dimensional integral).
- `F(c) = −kT log Z(c)` is the **free energy**. Unlike a single minimum energy, `F` accounts
  for *both* how deep the valleys are (enthalpy) *and* how many/wide they are (entropy). It is
  the physically correct quantity for comparing whole molecules.

**Why this is the real goal.** Generating a single low-energy structure is easy and not very
useful. The useful, hard objects are **equilibrium ensembles** and **relative populations**:
which conformers are actually occupied and in what ratio; which tautomer/protonation state
dominates; relative free energies (`ΔF`), binding, `pKa`. All of these are exactly what the
Boltzmann distribution (and `F`) encode, and exactly what ordinary data-fit generators do
**not** give you.

## 5. Boltzmann generators

A **Boltzmann generator** (Noé et al., *Science* 2019) is a generative model trained so that
its samples follow `π ∝ exp(−E/kT)` rather than an empirical data distribution. Classic BGs
are normalizing flows that provide an exact likelihood and are corrected to *exact* Boltzmann
by **importance reweighting** (weights `w = exp(−E/kT)/p_model`; quality is measured by
**effective sample size, ESS**). Most BGs are per-system, peptide-scale, and use classical
force fields. Our work pushes this to a **transferable, universal-MLIP, whole-periodic-table,
de-novo** setting, and avoids a fragile global reweighting.

## 6. Flow matching (our generative backbone)

We build on **flow matching (FM)**, the current SOTA family for continuous generative
modeling (and the basis of FlowMol3, which we extend).

- Draw a noise sample `x_0 ~ N(0, σ²I)` and pair it with a data point `x_1`. Define a straight
  **interpolant** `x_t = (1−t) x_0 + t x_1` for `t ∈ [0,1]`. The **velocity** that carries
  noise to data is `v = x_1 − x_0`. FM trains a network `v_θ(x_t, t)` to regress this velocity;
  sampling integrates the ODE `dx/dt = v_θ` from `t=0` (noise) to `t=1` (data).
- **Score–velocity identity.** The *score* of the intermediate marginal, `s(x_t,t) =
  ∇_{x_t} log p_t(x_t)`, is an affine function of the FM velocity. For the interpolant above,
  ```
  s(x_t, t) = (t · v_θ − x_t) / ((1−t) · σ²).
  ```
  (This is verified — sign and factors — in `notes/appendix_hbc.tex`; it diverges at `t=1`, so
  we evaluate near, not at, the data.) **This identity is the bridge from a flow model to a
  density gradient**, and therefore to forces/energies.
- **Exact likelihood (FFJORD).** A continuous flow admits an exact log-density via the
  instantaneous change-of-variables: integrate the velocity backward from `x_1` to `x_0` and
  accumulate the divergence, `log p_1(x_1) = log p_0(x_0) − ∫ ∇·v dt`. We use this to get
  `log p_θ(r|c)` at evaluation time (and to read out `F(c)`; see the theory).

FlowMol3 additionally handles **discrete channels** (atom types, charge) with a continuous-time
Markov chain (CTMC / discrete flow). So the full model jointly flows coordinates *and*
composition — which is what makes the two-level factorization below natural.

## 7. Conformer-generation metrics (and their limits)

The standard 3D-conformer benchmark (GEOM-DRUGS/QM9) measures how well generated conformers
**geometrically cover** a reference ensemble:

- **COV (Coverage), Recall & Precision** — fraction of reference (resp. generated) conformers
  that have a close match (RMSD < δ; δ=0.75 Å DRUGS, 0.5 Å QM9), in %.
- **AMR / MAT (Average Minimum RMSD)** — the average best-match RMSD, in Å.

These measure *geometry coverage*, not *thermodynamics*. They say nothing about whether the
**populations** are right, and the community has flagged them as chemically unreliable
(GEOM-Drugs-Revisited, 2025). This is precisely the gap our Boltzmann-fidelity metrics fill.

---

# Part II — Motivation

**The problem.** De-novo 3D molecular generation is central to drug discovery, catalysis, and
materials. But two things are missing from current generators:

1. **They fit data, not physics.** Diffusion/flow generators (EDM, GeoLDM, MiDi, ET-Flow,
   NExT-Mol, …) reproduce the distribution of a *dataset*. They do not target the Boltzmann
   distribution, cannot report relative populations or free energies, and inherit dataset bias.
2. **They are organics-only and bond-supervised.** Most assume a bond graph (a human
   abstraction that fails on TMs/radicals/hypervalents) and cover ~5–10 elements.

Meanwhile, the two "physics-aware" camps are **disjoint**:

- **Boltzmann/energy-aware conformer samplers** (Adjoint Sampling, Torsional-GFN) *are*
  Boltzmann-correct — but only for the conformation of a **fixed, given molecular graph**.
  They never generate the molecule itself.
- **Hierarchical de-novo generators** (NExT-Mol) *do* pick identity then geometry — but are
  pure data-likelihood, with **no energy, no Boltzmann target, no free energy**.

**Our thesis.** Occupy the empty intersection: a generator that is (a) **hierarchical**
(picks composition, then conformation), (b) **Boltzmann-correct at both levels** under a
**universal MLIP**, (c) **bond-free** and whole-periodic-table, and (d) linked across levels
by a **conditional partition function**. This is unclaimed, physically principled, and yields
capabilities (equilibrium populations, per-molecule free energies, exotic chemistry) no
baseline has.

---

# Part III — Method

## 8. The hierarchical factorization

Factor the joint generator as
```
p_{θ,φ}(c, r) = p_φ(c) · p_θ(r | c)
```
- `p_θ(r | c)` — **conformation level**: a *continuous flow-matching* model over coordinates
  with the molecular identity held fixed (bond-free). This is where forces/energies live and
  where flow matching is SOTA.
- `p_φ(c)` — **composition level**: a generator over the discrete identity (atom types, counts,
  charge). Naturally an autoregressive graph/SELFIES model or a discrete flow (CTMC). Its job
  is to make common molecules the *thermodynamically favourable* ones.

## 9. Level 2 — conditional conformation via on-policy MLIP distillation

We want `p_θ(r|c) = π(r|c) ∝ exp(−E(r|c)/kT)`. The lever is the score–velocity identity:

- **Force = score, up to `kT`.** At equilibrium `∇ log π(r|c) = −∇E/kT = F/kT`. So if we push
  the model's FM-implied score `s_θ = (t v_θ − x_t)/((1−t)σ²)` toward the teacher force
  `F/kT`, we are training the model's density to be locally Boltzmann. This is the **force
  loss** `L_force = ‖s_θ − F/kT‖²` (matching to `+F/kT` is Boltzmann-*attracting* — verified).
- **Off-policy vs on-policy.** The naive version evaluates the force at the *data* geometry
  (off-policy). We instead do **on-policy distillation**: let the current model *generate* a
  conformer, query the teacher for the force *at that generated geometry*, and match there.
  This closes the train/test gap (the model is corrected where it actually samples, not only
  where the data sits) and is the modern, more effective approach (cf. Adjoint Sampling).

Mechanically (per step, amortized): freeze the discrete channel to the molecule's identity,
roll out a coordinates-only Euler flow to get `x_gen` (stop-gradient), call the teacher for
`F_teacher` at `x_gen`, then take a gradient step matching `s_θ` to `F_teacher/kT` on the
conditional path toward `x_gen`.

## 10. The bridge — conditional free energy `F(c)`

The two levels are joined by the **conditional partition function** `Z(c) = ∫ exp(−E(r|c)/kT) dr`
and its free energy `F(c) = −kT log Z(c)`. The key, load-bearing fact:

> If the conformation model is Boltzmann-consistent, then for *any* conformer `r`,
> `F(c) = E(c,r) + kT · log p_θ(r|c)`, a quantity **independent of `r`**.

So `F(c)` is a **free by-product** of a good conformation model: evaluate the teacher energy
`E` (teacher) and the model's exact log-density `log p_θ` (FFJORD) at one (or a few) generated
conformers, and average. Its residual variance across conformers is itself a certificate of
how Boltzmann-consistent the conformation model is. (This is exactly the quantity the earlier
within-parent variance loss *discarded*; we now recover and use it.)

## 11. Level 1 — free-energy-weighted composition prior

We then train the composition model so that
```
p_φ(c) ∝ exp(−F(c)/kT)
```
i.e. molecules are generated in proportion to their (Boltzmann) equilibrium weight. Because
`F(c)` is a scalar per molecule, this is a **reward-tilting** problem (GFlowNet /
reward-weighted training), with the reward `R(c) = −F(c)/kT` supplied entirely by Level 2 +
the teacher. The loop closes without ever forming the intractable grand partition function.

## 12. Implementation

- **Backbone:** FlowMol3, extended (not forked) via runtime **monkey-patches**. Layer 1
  (`cfm_mol/flow_model.py`) adds geometric-constraint hooks; Layer 2
  (`cfm_mol/bgfm_train_hook.py`) wraps `training_step` to add the physics losses.
- **Losses** (`cfm_mol/bgfm_loss.py`): `score_from_fm_velocity`, `force_loss` (match to
  `F/kT`), plus energy/anchor terms; FFJORD log-density and `F(c)` readout in
  `cfm_mol/bgfm_density.py`.
- **Teacher:** the eSEN model runs as an XMLRPC worker (`scripts/omol25_worker.py`, omol25
  env); the trainer (flowmol env) queries it on-policy via `cfm_mol/physics_drift.py`.
- **Data:** OMol25 → FlowMol-native `.pt` with **DFT forces + energies stored**
  (`scripts/preprocess_omol25.py`); the dataset wires `force_1_true` into the graph so the
  physics loss has a target. (This wiring was a critical fix — see §18.)
- **Two conda envs:** `envs/flowmol` (torch 2.2 + DGL, training/inference) and `envs/omol25`
  (torch 2.8 + fairchem, preprocessing + teacher). Kept separate on purpose.

Implementation status: **Level 2 (conditional conformation + on-policy distillation) is
implemented**; the `F(c)` readout and **Level 1 (composition prior)** are the next phase.

---

# Part IV — Theoretical analysis

Full statements/proofs in `notes/appendix_hbc.tex`; the ideas:

## 13. Target and factorization = statistical mechanics

The joint Boltzmann `π(c,r) ∝ exp(−E(c,r)/kT)` factors exactly as `π(c,r) = π(c)·π(r|c)` with
`π(r|c) = exp(−E/kT)/Z(c)` and `π(c) = Z(c)/𝒵 = exp(−F(c)/kT)/𝒵`, `𝒵 = Σ_c Z(c)`. This is
**not a modeling choice** — it is the decomposition of the molecular partition function into
an *intramolecular* configurational integral `Z(c)` and an *inter-species* chemical sum `𝒵`.
Our two model factors are in one-to-one correspondence with these physical factors.

## 14. Hierarchical Boltzmann Consistency (main theorem)

`p_{θ,φ} = π` **iff** (i) `p_θ(r|c) = π(r|c)` for all `c`, **and** (ii) `p_φ(c) ∝ exp(−F(c)/kT)`.
Elementary as an identity — its force is that the two conditions are *separately trainable*
and that condition (ii) is governed by the otherwise-intractable free energy `F(c)`, which
(Corollary, §10) is recoverable from the conformation model + teacher.

## 15. Gauge-identifiability (why the hierarchy is *necessary*)

**Theorem.** From forces alone, the *conditional* Boltzmann `π(r|c)` is fully identified, but
the *joint* `π(c,r)` is identified **only up to a per-composition free-energy gauge** `{F(c)}`:
adding any function `g(c)` of composition alone leaves every force and every conditional
unchanged while reweighting the joint by `e^{−g(c)/kT}`. Consequence: **no force-only / flat
model (flat force-matching, denoising energy matching, adjoint samplers) can ever be globally
Boltzmann** — the undetermined degrees of freedom are exactly the free energies `{F(c)}`. The
two-level hierarchy is the *minimal* parameterization that isolates that gauge into `p_φ(c)`.
This also explains why BGFM needs **both** a force term (fixes the conformation level) **and**
an energy/`F(c)` term (fixes the composition gauge): forces train shape, energies train weight.

## 16. Modular guarantee

The chain rule gives an **exact, additive** error decomposition:
```
KL(π ‖ p) = KL(π(c) ‖ p_φ(c)) + E_{c∼π}[ KL(π(r|c) ‖ p_θ(r|c)) ].
```
No cross-term. Train each level to `ε` and the joint is within `ε_c + ε_r`. Moreover the
composition target `F(c)` is estimable from the (already-trained) conditional model with bias
controlled by *its* error — so the two errors are **independently certifiable** and the bridge
adds no uncontrolled third error. This turns "provably Boltzmann" from an idealized slogan into
a finite, decomposable bound tied to two trainable losses.

## 17. Efficiency (variance collapse)

Correcting a proposal to the *joint* Boltzmann by importance sampling has weights whose
variance grows **exponentially in `3N`** (this is why all-atom Boltzmann generators report
ESS of 0.04–0.5). Under the hierarchy the conditional level is natively Boltzmann (no
reweighting), and the only correction is the composition tilt by the **scalar** `F(c)` — a
one-dimensional reweighting per molecule, variance independent of `3N`. The hierarchy converts
an exponential-variance problem into a bounded-variance one; this is why there is no ESS collapse.

*(Each of §13–17 is specific to molecular Boltzmann generation — the gauge is the chemical
free energy, the scale separation is intra- vs inter-molecular — and none is available to any
prior flat, guidance, or single-level method. Together they make the oral-level case:
factorization dictated by physics + a no-go that forces the hierarchy + a modular guarantee +
an efficiency argument.)*

---

# Part V — Experimental setting

Numbers, per-cell values, and the fairness audit live in `notes/benchmark_plan.md`.

## 18. Data & scale

- **Train:** OMol25 (bond-free, 83 elements, DFT forces+energies). Subsets: 50k (sanity/dev),
  4M (headline), 100M available.
- **Transfer / OOD eval:** GEOM-DRUGS & GEOM-QM9 (zero-shot conformer transfer — we do *not*
  train on GEOM); planned tmQM / kraken / hypervalent for the wide-chemistry "killer figure."
- **Caveat, stated plainly:** small (50k) checkpoints are undertrained and not competitive; the
  headline needs the 4M model. And **all `.pt`-based "BGFM" runs before 2026-07-02 accidentally
  trained with the physics loss disabled** (a data-wiring bug: `force_1_true` never reached the
  graph, so the hook silently trained plain FM). That is fixed (forces now wired, loud guard
  added, all sign conventions adversarially verified correct); pre-fix numbers are physics-free
  and not to be trusted as BGFM.

## 19. What we measure (metrics) and why

| Metric | What it tests | Why it matters here |
|---|---|---|
| Validity / connectivity / **PoseBusters** (all-checks) / uniqueness | basic chemical sanity of generated 3D molecules | directly comparable to the OMol25 generator Zatom-1 (same definitions) |
| **COV-R/P, AMR-R/P** (GEOM) | geometric coverage of conformer ensembles | the standard leaderboard; our zero-shot-transfer row shows non-regression |
| GFN2-xTB Boltzmann-weighted **ensemble-property error** | physical quality of the ensemble | bridges COV/MAT to thermodynamics |
| **Boltzmann-R²** (`log p_θ` vs `−E/kT`) | does the density track the Boltzmann energy? | our core claim; scored against an **independent** potential (xTB), *not* the eSEN we train on, to kill circularity |
| **ESS without reweighting**, **ΔF error**, relative-population accuracy | thermodynamic fidelity / free energy | the differentiator — most baselines cannot even compute these |
| **xTB ΔE / atom** (relaxation) | model-agnostic physical quality | the only metric fair across all model families (it uses no model's log-p, and xTB is in no training loss) |

## 20. What we compare against, and why each

Comparability rule: **numbers are only comparable within a shared protocol** (dataset, split,
coverage threshold, sample budget, metric definition). Hence several tables, never one.

- **Conformer-generation SOTA — GeoDiff, GeoMol, Torsional Diffusion, MCF, ET-Flow, DiTMC,
  AvgFlow, Energy-Guided-FM.** *Why:* the established GEOM COV/MAT leaderboard. We join it via
  **zero-shot transfer** to demonstrate we are competitive on standard geometry metrics despite
  training on a totally different (OMol25, bond-free) distribution. We do **not** expect to win
  here (COV/MAT is saturating and criticized) — the point is non-regression.
- **On-policy / Boltzmann conformer samplers — Adjoint Sampling, Torsional-GFN.** *Why:* these
  are the **closest prior art** to our conformation level (on-policy MLIP distillation; Boltzmann
  conformers). We must show we recover their conditional-conformer quality *and* add the
  composition level + free-energy bridge they lack. (Note Adjoint Sampling reports at δ=1.25 Å —
  not directly comparable to the 0.75 Å leaderboard, so it's listed separately.)
- **Boltzmann generators — Transferable BG, Sequential BG, PROSE, FAB, iDEM.** *Why:* they
  define the **free-energy / ESS** capability and metrics. They are peptide/particle-scale with
  classical force fields, so this is a **capability comparison** (can we report ESS/ΔF at all,
  on far broader chemistry?), not a head-to-head leaderboard.
- **De-novo 3D generators — EDM, GeoLDM, MiDi, JODO, NExT-Mol, and especially Zatom-1.** *Why:*
  Zatom-1 is the **only external de-novo generator on OMol25** (our home turf) — the most direct
  baseline. It is bond-free like us but pure likelihood (no physics) and non-converged; matching
  its validity/PoseBusters definitions makes this an apples-to-apples home-field comparison. The
  others (EDM/GeoLDM/MiDi/JODO/NExT-Mol) are QM9/GEOM-home-field context and mostly bond-supervised.

## 21. The comparison tables (structure)

1. **Table 1 / 1b — GEOM-DRUGS / QM9 conformer COV/MAT** (shared leaderboard; our row = zero-shot).
2. **Table 1c — GEOM xTB Boltzmann-weighted ensemble properties** (physical conformer quality).
3. **Table 2 — de-novo generation on OMol25** (home turf; vs Zatom-1; validity/PoseBusters/
   uniqueness + xTB ΔE + Boltzmann-R²).
4. **Table 3 — Boltzmann / free-energy fidelity** (Boltzmann-R², ESS-without-reweighting, ΔF) —
   the differentiator; most baselines are N/A.
5. **Table 4 — capability matrix** (bond-free / universal-MLIP / de-novo-composition / exact
   log-p / Boltzmann-target / free-energy / on-policy / whole-periodic-table) — the oral
   centerpiece; only our row is all-✓.

Baseline cells are **source-verified** (adversarial re-fetch of the papers) and fairness-audited;
our rows fill from the automated eval chain when the checkpoints land.

## 22. Ablations & what "success" looks like

- **Ablations:** force-only vs +on-policy vs +energy/`F(c)`; off-policy vs on-policy; with/without
  the composition prior; kT-conditioning; negative-control (shuffled-atom force target).
- **Oral-worthy claims:** (a) a Boltzmann-fidelity result (positive Boltzmann-R² against an
  independent potential, competitive ESS **without** reweighting) that conformer SOTA and Zatom-1
  cannot produce; (b) **relative populations** across tautomer/protonation/constitutional isomers
  within DFT error — a claim COV/MAT literally cannot express; (c) **wide-chemistry** coverage
  (TM/radical/hypervalent) where organics-only baselines are untested; (d) the capability matrix
  where we are the only all-✓ row — backed by the theory in Part IV.

---

## Appendix — file map

- `notes/PROJECT.md` — this document.
- `notes/appendix_hbc.tex` — formal theory (theorem, gauge-identifiability, guarantees).
- `notes/benchmark_plan.md` — comparison tables (verified baseline numbers), fairness audit,
  eval-path/scripts status, current results, and the physics-no-op bug/fix writeup.
- `~/.claude/plans/recap-me-with-thi-merry-lagoon.md` — the master plan (phasing).
- `cfm_mol/` — method code (`bgfm_train_hook.py`, `bgfm_loss.py`, `bgfm_density.py`,
  `physics_drift.py`, `flow_model.py`).
- `scripts/` — preprocess, training launchers, eval (`eval_boltzmann_independent.py`,
  `eval_geom_covmat.py`, `benchmarks/omol25/validity_from_json.py`), SLURM in `scripts/unity/`.
