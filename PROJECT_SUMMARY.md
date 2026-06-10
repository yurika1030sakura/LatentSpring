# BGFM — Project Summary (as of 2026-06-09)

**Boltzmann-Guided Flow Matching (BGFM): a universal-chemistry 3D molecular
generator whose learned density is pushed to match the physical Boltzmann
distribution under the OMol25 neural potential.**

Target venue: ICLR 2027. This file is a self-contained snapshot of the
**current winning state** and **what goes into the paper**. Repo:
`/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/`.

---

## 1. One-line pitch

Most 3D molecular generators only *imitate* a dataset — they produce
molecules that look valid but are not physically equilibrated, and they
collapse outside the chemistry they were trained on. BGFM trains a flow
matching generator so that **how likely it thinks a shape is** matches
**how stable physics says that shape is** (the Boltzmann law), using a
single *universal* neural potential (OMol25, 83 elements) as the physics
signal. The payoff: generated geometries are closer to physical
equilibrium (cheaper downstream DFT/MD) and the model works across organic,
organometallic, and transition-metal chemistry from one trained model.

---

## 2. The task (plain)

Generate a new molecule as a 3D cloud of labeled atoms: for each of `N`
atoms, a **position** (x,y,z) and a **type** (which of 83 elements), plus a
per-atom charge. No bonds are generated (see §4). `N` is drawn from the
empirical size distribution before generation; the flow then positions and
labels exactly `N` atoms. Generation is **unconditional**.

---

## 3. Data: OMol25

- Source: OMol25, a public dataset of molecular geometries each labeled by a
  high-quality quantum-chemistry calculation. Training cut: 4M structures
  (full set 100M).
- Each record provides: atom positions, atom elements (83-element vocab),
  total charge, a scalar **energy** `E`, and a per-atom **force** `F`
  (the downhill direction of energy). No bonds, no SMILES.
- Preprocessing packs these into FlowMol-native tensors and precomputes
  marginal element/charge distributions and the molecule-size histogram.
  It also writes **perturbation shards**: for each parent molecule, K small
  geometric jiggles, each re-labeled with its OMol25 energy — these feed the
  energy loss (§5).
- Train / val / test split is standard; the headline metric is evaluated on
  **held-out** molecules.

---

## 4. Method

### 4.1 Base model
Fork of **FlowMol3** (joint discrete+continuous 3D flow matching, SE(3)-
equivariant GVP-Transformer). The architecture, sampler, and data loaders
are left intact; the contribution is a training-step hook plus a data
pipeline.

- **Continuous channel** (positions): Gaussian prior, linear interpolant,
  velocity-MSE flow matching.
- **Discrete channels** (element, charge): start from an "unknown"
  placeholder and resolve to a real class during the flow (discrete flow
  matching).

### 4.2 Bond-free
Bonds are **not** generated (`total_loss_weights.e = 0`). OMol25 has no bond
labels (quantum chemistry has no bond orders), so forcing bond prediction
would teach wrong labels, especially on metals. Bonds, if needed, are
inferred post-hoc from the generated geometry. This sharpens the story:
the model generates physical reality (atom positions); bonds are an
afterthought.

### 4.3 The loss

```
L_total = L_FM + λ₁·L_force + λ₂·L_energy
```

- **L_FM** — standard flow matching: velocity MSE on positions +
  cross-entropy on element/charge. Fits the data density.
- **L_force** — the model's *implied gradient of log-density* should equal
  the physics force `F/kT`. The local (slope) form of Boltzmann.
- **L_energy** — the model's *log-density* should match `−E/kT` (up to a
  constant). The global (value) form of Boltzmann.

The three terms are redundant only in the infinite-data/infinite-capacity
limit; in practice each gives a different learning signal (dense velocity
fit / local gradient / global shape).

### 4.4 The tricks that make it trainable

A naive Boltzmann term is brutal — it needs the model's likelihood (an ODE +
divergence integral with backprop), the intractable partition function `Z`,
and would backprop through the physics model. Three moves remove these:

1. **Frozen physics labels.** `E` and `F` are precomputed and stored; during
   training they are constant regression targets. The OMol25 model is never
   in the computational graph — no backprop through the potential.

2. **Score-from-velocity closed form.** The force term reads the implied
   score directly off a single-time velocity by algebra,
   `s = (t·v − x) / ((1−t)·σ²)`, evaluated at multiple late times. No ODE
   integration, no trajectory backprop, no `Z`. Cheap; runs every step.

3. **Variance form for the energy term.** Instead of hitting the unknown
   constant `−log Z`, require `log p + E/kT` to *be* constant by minimizing
   its variance **within each molecule** across its K perturbations. The
   within-molecule grouping cancels the per-molecule `Z`. An optional small
   invariant "log-Z predictor" anchors the absolute level to prevent a
   trivial constant solution.

The residual cost is the true log-density in the energy term (an FFJORD
trajectory-divergence estimate). It is made affordable with a Hutchinson
divergence estimator, a shallow ODE (~4 steps), a tiny batch, and
amortization (every 4th step). The energy term is a light auxiliary nudge
(`λ₂` small); the force term carries the cheap per-step signal.

---

## 5. Current results (winning state)

Headline metric — **per-molecule perturbation R²**: jiggle one held-out
molecule into many shapes, ask the model how likely each is (`log p`) and
physics how stable each is (`−E/kT`), and measure how linearly the two
track. R²→1 = the model ranks shapes exactly by physical stability;
slope→1 = quantitatively matches the Boltzmann law.

Best checkpoint (v7c, step 50k):

| Slice | R² | slope | frac r>0.5 |
|---|---|---|---|
| Overall (≤50 atoms) | **0.278** | 0.840 | 67.5% |
| Organic (CHNOFS) | **0.380** | 0.800 | 65% |
| **Transition metal** | **0.501** | 1.40 | **80%** |

- vs the flow-matching-only baseline R² = 0.091 → **~3× tighter** alignment
  with physics, with slope 0.84 vs 0.29.
- The **strongest slice is transition metals** — the strategically important
  case, since no prior physics-based molecular generator operates on metals
  at all. The "universal chemistry" claim is empirically validated where it
  matters most.
- Training healthy: energy loss dropped ~9× and force-alignment rose ~5×
  over training, numerically stable end-to-end.

---

## 6. What goes into the paper

### 6.1 Contributions
1. **Method:** a per-molecule within-perturbation energy-variance loss (with
   an invariant log-Z anchor) that aligns a flow model's density to a
   Boltzmann target *inside a single FM training loop* — the first
   energy-consistency formulation that works on a universal neural potential.
2. **System:** an offline OMol25 perturbation-precompute pipeline + paired
   two-environment training setup that scales energy-consistency training to
   large datasets without the physics-evaluation cost overwhelming training.
3. **Empirical:** the first universal-NP-guided Boltzmann generator for 3D
   molecules, with a new Boltzmann-correlation metric and (planned)
   downstream-deployment numbers across 83 elements.

### 6.2 Experiment plan
- **Level 1 — Boltzmann correlation (done, headline):** per-molecule R² on
  held-out molecules, vs flow-matching baseline. **R² = 0.278 vs 0.091.**
- **Level 2 — downstream relaxation (planned):** how many fewer
  energy-minimization steps BGFM geometries need to reach equilibrium vs
  baseline geometries. Direct "saves downstream compute" table. Scripts
  ready.
- **Level 3 — sampling efficiency (planned):** whether a few-step BGFM
  ensemble matches a long molecular-dynamics ensemble's statistics, at a
  fraction of the compute. Scripts ready.
- **Coverage demonstration:** validity + R² per chemistry slice (organic /
  transition-metal / heavy main-group / charged), showing one model spans
  the 83 elements.
- **Ablation matrix:** FlowMol baseline / bond-free / OMol25+FM-only /
  full BGFM — isolates the bond-free variable and the BGFM losses.

### 6.3 Positioning (related work)
- **3D molecular generators (EDM, MiDi, FlowMol3):** optimize sample-set
  validity; never enforce Boltzmann.
- **Boltzmann generators (Noé, Klein-Noé, Scalable BG):** target Boltzmann
  but for a *fixed classical potential on a single system* (alanine, water,
  one protein). No cross-chemistry generalization.
- **BGFM's gap-filling claim:** universal chemistry (any element) +
  Boltzmann density (deployable starting points) + one trained model.

### 6.4 Figures / tables (main text target: 8 figs, 5 tables)
- Concept figure (NP + FM + 3 losses); per-molecule loss diagram.
- R² across baselines and across chemistry slices (universal claim).
- Downstream relaxation step distribution; BGFM-vs-MD energy histograms.
- Ablation bar chart; training-trajectory diagnostics.
- Prior-Boltzmann-generator comparison table (coverage / single-system /
  fixed-potential); compute-cost table.

---

## 7. Theory (honest framing)

The appendix states a **Boltzmann-consistency** result: in the idealized
limit (infinite capacity/data, energy weight → ∞) the global optimum of the
objective is exactly the Boltzmann distribution restricted to the data
support. This is a **consistency/sanity result** — it shows the objective
points at the right target — not a guarantee about the finitely-trained
model. The score-from-velocity relation used by the force term is a clean,
standard identity for the Gaussian interpolant. The paper's weight rests on
the **empirical** results, not on the theory.

---

## 8. Limitations (stated in the paper)
- **Boltzmann w.r.t. OMol25, not DFT** — physical accuracy is bounded by the
  neural potential's accuracy.
- **Single temperature** — trained at one `kT`; other temperatures need a
  conditioned variant (built but not the headline).
- **Finite perturbation precompute** — the energy loss uses an offline
  subset; generalization from it is measured but is the main attackable
  point.
- **No explicit bonds** — connectivity is derived post-hoc.
- **Memory ceiling** — FFJORD backward caps batch size.

---

## 9. Status & remaining critical path
- ✅ Level 1 headline result (R² = 0.278, ~3× baseline; best on metals).
- ⏳ Confirm the checkpoint sweet spot (eval neighboring steps) + multi-seed.
- ⏳ Level 2 (relaxation savings) — ~1 GPU-day, scripts ready.
- ⏳ Level 3 (sampling efficiency vs MD) — ~1–2 weeks, scripts ready.
- ⏳ Per-slice coverage table; baseline generator retrained on OMol25 for a
  head-to-head.

---

*Snapshot compiled 2026-06-09. Reflects the current winning configuration
(force + energy BGFM, bond-free, OMol25 4M) and the intended paper. For the
live working notebook with day-to-day numbers see the project memory file
`bgfm_results_and_novelty.md`; for the method derivation see
`bgfm_method.md`; for the full paper draft scaffold see
`paper_v2_bgfm_draft.md`.*
