# Boltzmann-Guided Flow Matching (BGFM): Method Derivation

**Version 1, 2026-04-21.** Paper 1 upgraded formulation.

This document contains the full mathematical derivation for BGFM, the
target method for our ICLR paper. Companion: `notes/section3_bgfm.md`
(main-text writeup), `notes/appendix_A_v4.tex` (formal proofs).

---

## 1. Setup

### 1.1 Configuration space

As in Section 3 bond-free formulation:
$$
x = (r, \tilde a, \tilde c) \in \mathcal{X} = \mathbb{R}^{3N} \times (\Delta^{A-1})^N \times (\Delta^{C-1})^N
$$
Positions, atom-type simplex, charge simplex. Bond-free: no edge/bond state.

### 1.2 Target distribution

Let $p_\text{data}$ denote the empirical OMol25 4M molecular distribution.
Let $E : \mathcal{X} \to \mathbb{R}$ denote the OMol25 neural potential
energy function, $C^1$ on a neighborhood of $\text{supp}(p_\text{data})$.
Let $F(x) = -\nabla_r E(x) \in \mathbb{R}^{3N}$ be the force field (only
the position coordinate is affected).

Define the **Boltzmann distribution** at temperature $kT$:
$$
p_\text{Boltz}(x) := \frac{1}{Z}\exp\!\big(-E(x)/kT\big), \qquad Z = \int \exp(-E(x')/kT)\, dx'.
$$

### 1.3 Goal

Train a flow matching model $v_\theta : \mathcal{X} \times [0, 1] \to T\mathcal{X}$
such that the induced density $p_\theta(x, t = 1)$ satisfies:

**(Condition 1, data match)** $p_\theta \approx p_\text{data}$ on the
training support.

**(Condition 2, force consistency)** $\nabla \log p_\theta(x) = -\nabla E(x)/kT$
on the training support.

**(Condition 3, energy consistency)** $\log p_\theta(x) = -E(x)/kT + \text{const}$
on the training support.

The three conditions are not independent: (3) implies (2) by taking gradients.
But enforcing them as **separate loss terms** gives different learning
signals and different failure modes, so we train all three jointly.

---

## 2. Flow matching foundation

Let $p_0 = \mathcal{N}(0, I) \otimes \text{Unif}(\Delta^{A-1})^N \otimes \text{Unif}(\Delta^{C-1})^N$
be the base distribution. Define the conditional interpolant:
$$
x_t = (1-t) x_0 + t x_1, \quad x_0 \sim p_0, \; x_1 \sim p_\text{data}.
$$
The **target marginal velocity** is
$$
u^*(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x].
$$
The **FM loss** is
$$
\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t, x_0, x_1}\big[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\big].
$$
By Lipman et al. 2023, $v_\theta \to u^*$ implies
$p_\theta(\cdot, t) \to p_t := (1-t) p_0 + t p_\text{data}$ (pointwise).

---

## 3. Score derivation from flow matching velocity

For flow matching with linear interpolant and Gaussian base $p_0$,
the **marginal score** at time $t$ relates to the marginal velocity $u^*$ via:
$$
s^*(x, t) = \nabla_x \log p_t(x) = -\frac{x - t \cdot \mathbb{E}[x_1 \mid x_t = x]}{(1-t)^2}
\qquad (\text{for } t < 1).
$$

**Derivation.** The conditional density is
$p_t(x \mid x_1) = \mathcal{N}(t x_1, (1-t)^2 I)$, so
$\nabla_x \log p_t(x \mid x_1) = -(x - t x_1) / (1-t)^2$.
The marginal score is the expected conditional score:
$\nabla_x \log p_t(x) = \mathbb{E}[\nabla_x \log p_t(x \mid x_1) \mid x_t = x]$.

In terms of the velocity: since $v^*(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]
= \mathbb{E}[x_1 \mid x_t = x] - \mathbb{E}[x_0 \mid x_t = x]$
and $x_t = (1-t) x_0 + t x_1$ gives
$\mathbb{E}[x_0 \mid x_t] = (x_t - t \mathbb{E}[x_1 \mid x_t]) / (1-t)$,
solving yields:
$$
\boxed{\quad s^*(x, t) = \frac{-1}{(1-t)}\Big(x - t \mathbb{E}[x_1 \mid x_t = x]\Big) / (1-t) - \frac{v^*(x, t)}{1-t}\quad}
$$

For the **learned** model:
$$
s_\theta(x, t) := -\frac{1}{(1-t)}\Big(x - t \big(x + (1-t) v_\theta(x, t)\big)\Big)/(1-t) - \frac{v_\theta(x, t)}{1-t}.
$$

In the $t \to 1$ limit, careful handling: we evaluate score at $t_\text{eval}$
close to $1$ (e.g., $t_\text{eval} = 0.99$) to avoid division by $(1-t) \to 0$.

**Remark.** The exact score formula depends on the prior; this is the
Gaussian-prior case. For simplex channels we derive separate formulas;
see Appendix A v4 Section~4 for the CTMC-discrete score.

---

## 4. Force consistency loss

For $x \sim p_\text{data}$ and $t = t_\text{eval}$ (close to 1), enforce
the score to match the Boltzmann force:
$$
\mathcal{L}_\text{force}(\theta) = \mathbb{E}_{x \sim p_\text{data}}\Big[\|s_\theta(x, t_\text{eval}) - F(x)/kT\|^2\Big],
$$
where $F(x)/kT$ is the **precomputed** scaled force from OMol25
(computed once in the offline preprocessing stage).

**Implementation note.** At data points $x_1 \sim p_\text{data}$,
the conditional velocity is $(x_1 - x_0)$ where $x_0$ is a random draw
from the prior. To avoid the $(1-t) \to 0$ singularity, sample
$t_\text{eval} \sim \text{Unif}[0.9, 0.99]$ and weight the loss
accordingly.

---

## 5. Log-density via Hutchinson trace

The log-density at $t = 1$ is obtained via the flow-based
change-of-variables formula:
$$
\log p_\theta(x_1) = \log p_0(x_0) - \int_0^1 \text{div}\big(v_\theta(x_t, t)\big)\, dt,
$$
where $x_0$ is the pullback along the learned ODE from $x_1$.

The divergence can be estimated by **Hutchinson's trace estimator**:
$$
\text{div}(v) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)}\big[\xi^\top J_v \xi\big],
$$
where $J_v$ is the Jacobian of $v$ w.r.t. $x$. The inner product
$\xi^\top J_v \xi$ is efficiently computed by **vector-Jacobian product**:
$\xi^\top J_v \xi = \xi^\top \partial_x(v \cdot \xi)$, available in PyTorch
via `torch.autograd.grad`.

To keep training tractable, we use:
- **Stochastic trace**: $K = 1$ Hutchinson sample per step (low variance due to large batch)
- **Euler trajectory**: evaluate divergence at $M = 10$ equally-spaced
  checkpoints along the trajectory, integrate via trapezoidal rule
- **Variance reduction**: Rademacher $\xi \in \{-1, +1\}^d$ instead of
  Gaussian reduces estimator variance by factor $\sim 2$

### Efficient implementation via exact trace
Since our model is GVP-based, the Jacobian is block-diagonal over atoms
(each atom's output depends only on its neighbors). Exact trace is
$\mathcal{O}(N)$ rather than $\mathcal{O}(N^2)$ if done atom-by-atom:
$$
\text{div}(v) = \sum_{i=1}^N \text{tr}\big(\partial_{r_i} v_{r_i}\big).
$$
For moderate $N \le 200$ this is **exact** and feasible, avoiding
Hutchinson noise entirely.

---

## 6. Energy consistency loss

For $x_1 \sim p_\text{data}$ (endpoint), enforce:
$$
\mathcal{L}_\text{energy}(\theta) = \mathbb{E}_{x_1 \sim p_\text{data}}\Big[\big(\log p_\theta(x_1) + E(x_1)/kT - C\big)^2\Big],
$$
where $C$ is a **learned per-batch constant** that absorbs the log
partition function. In practice, subtract the batch mean to center:
$$
\mathcal{L}_\text{energy}(\theta) = \mathrm{Var}_{x \sim \text{batch}}\big(\log p_\theta(x) + E(x)/kT\big).
$$

This variance formulation is **affine-invariant to the partition function**:
$p_\theta$ is consistent up to a constant, so only the shape of
$\log p_\theta(\cdot) - (-E(\cdot)/kT)$ matters.

---

## 7. Full training objective

$$
\boxed{\quad
\mathcal{L}_\text{total}(\theta) = \mathcal{L}_\text{FM}(\theta) + \lambda_1\, \mathcal{L}_\text{force}(\theta) + \lambda_2\, \mathcal{L}_\text{energy}(\theta).
\quad}
$$

### Hyperparameter schedule
- **Warmup**: first 10% of training, $\lambda_1 = \lambda_2 = 0$
  (pure FM, model stabilizes)
- **Ramp**: linear ramp $\lambda_1 : 0 \to 0.5$, $\lambda_2 : 0 \to 0.1$
  over 20% of training
- **Full**: remainder uses full weights
- Tune $\lambda_1, \lambda_2$ via QM9 grid search before scaling to 4M OMol25

### Loss weight rationale
- $\lambda_1 > \lambda_2$: force consistency is a **local** per-sample
  signal (fast gradient); energy consistency is a **global** per-sample
  signal that requires accurate Jacobian estimate (slower gradient,
  noisier)
- Warmup prevents early-training instability when model is far from FM
  optimum (Hutchinson estimator has large variance)

---

## 8. Boltzmann consistency theorem (Paper 1 core claim)

**Theorem (BGFM-Boltzmann consistency).** Let $\theta^*$ minimize
$\mathcal{L}_\text{total}$ as $\lambda_1, \lambda_2 \to \infty$ jointly,
subject to a ceiling $\mathcal{L}_\text{FM}(\theta^*) \le \varepsilon_\text{FM}$.
Then the induced density $p_{\theta^*}$ satisfies:
$$
p_{\theta^*}(x) = \frac{\exp(-E(x)/kT)}{Z^*} \quad\text{for all}\ x \in \mathrm{supp}(p_\text{data}),
$$
where $Z^* = \int_{\mathrm{supp}(p_\text{data})} \exp(-E(x')/kT)\, dx'$.

**Proof sketch.** As $\lambda_2 \to \infty$:
$\mathcal{L}_\text{energy} \to 0 \Rightarrow
\text{Var}_x(\log p_\theta + E/kT) \to 0
\Rightarrow \log p_\theta(x) + E(x)/kT$ is constant on
$\mathrm{supp}(p_\text{data})$
$\Rightarrow p_\theta(x) \propto \exp(-E(x)/kT)$.

The normalization $Z^*$ follows from $\int p_\theta = 1$ restricted to
$\mathrm{supp}(p_\text{data})$ (flow matching covers the support via
$\mathcal{L}_\text{FM} \le \varepsilon_\text{FM}$).

$\mathcal{L}_\text{force}$ provides a redundant but **local** consistency
check: it ensures that even in regions where the energy loss has not yet
converged (high-energy outliers), the local gradient structure of
$\log p_\theta$ matches Boltzmann. This improves sample quality on the
tail.

**Corollary (chemistry-agnostic Boltzmann).** Since $E$ is supplied by
the OMol25 universal neural potential (83-element coverage),
the Boltzmann consistency holds uniformly across organic, organometallic,
radical, and hypervalent chemistries — no chemistry-specific prior is
invoked.

**Full proof**: Appendix A v4, Theorem 8.

---

## 9. Sampling

After training, at inference we combine:
1. **Flow integration** via $v_\theta$ (no physics drift needed in
   principle — the model has learned Boltzmann internally).
2. **Optional** OMol25 drift for out-of-distribution correction
   (Level 1 mechanism, retained for OOD robustness).

Ablation: Level-2-only vs Level-1-only vs full.

---

## 10. Implementation checklist

### Preprocessing (Week 1, omol25 env)

```python
# For each training molecule x_1_true:
E_true = OMol25.compute_energy(x_1_true)    # scalar
F_true = OMol25.compute_forces(x_1_true)    # (N, 3); ~0 for equilibrium
store(x_1_true, E_true, F_true)
# Total: 4M mol * ~100ms = 7 CPU-hr on 16-worker cluster (~25 min wall)
```

Extend `train_data_processed.pt` schema:
- Add `energies`: (M,) float32 tensor, one per molecule
- Add `forces`: (N_total, 3) float32 tensor, concatenated per atom

### Code additions (Week 2, flowmol env)

New file `cfm_mol/bgfm_loss.py`:
- `score_from_fm_velocity(v, x, t)` — Eq. 3.1 of this doc
- `divergence_exact(v, x)` — atom-by-atom exact trace (GVP-compatible)
- `divergence_hutchinson(v, x, K=1)` — fallback for large $N$
- `bgfm_loss(batch, model, lambda_1, lambda_2)` — full 3-term loss

Modify `cfm_mol/flow_model.py::FlowMolL`:
- Accept `energies`, `forces` from batch
- Call `bgfm_loss` when BGFM config flag set
- Log individual loss components separately

Config (`configs/omol25_4m_bgfm.yaml`):
- `mol_fm.bgfm.lambda_1: 0.5`
- `mol_fm.bgfm.lambda_2: 0.1`
- `mol_fm.bgfm.warmup_frac: 0.1`
- `mol_fm.bgfm.ramp_frac: 0.2`

### Training (Week 3)

- Small-scale validation: QM9 subset, 500k steps, grid search
  $\lambda_1 \in \{0.1, 0.5, 1.0\}$, $\lambda_2 \in \{0.01, 0.1, 0.5\}$
- Full-scale: Plan B config, BGFM enabled, 6-8 H100-days

### Experiments (Week 4)

- Ablation A: vanilla FM (baseline)
- Ablation B: FM + force only
- Ablation C: FM + energy only
- Ablation D: FM + force + energy (BGFM full)
- Metric: TV distance to held-out data, log-density fit to Boltzmann,
  sample quality (max $|F|$, DFT relax steps), OOD generalization on
  tmQM/kraken

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Energy variance too noisy, training unstable | Medium | Use exact atom-by-atom trace (Section 5); gradient clipping; warmup |
| $\lambda_2$ tuning impossible (collapses or vanishes) | Medium | Grid search on QM9 before scaling; log-space tuning |
| Score formula wrong for simplex channels | Low | Cross-check via finite differences on toy data |
| Theorem has hidden assumption | Low-medium | Have a weaker fallback theorem ("local" Boltzmann on support) |
| $F_\text{OMol25}$ noisy near training geometries | Low | OMol25 is DFT-accurate; forces on equilibrium geometries are very close to 0 |
| Full training diverges on OMol25 | Low | Checkpoint resume + stage the loss (FM first, add force, add energy) |

---

## 12. Fallback plan

If the full BGFM (all three losses) does not converge, step down:

1. **Level 1.5 (force-only)**: $\mathcal{L}_\text{FM} + \mathcal{L}_\text{force}$.
   Still novel ("force-matched flow matching"). Publishable.
2. **Level 1 (original physics drift)**: revert to inference-time drift
   only. Full Plan B config. Still publishable.

At all fallback levels, the paper can still be submitted; we lose novelty
incrementally but never catastrophically.

---

## 13. Paper structure implication

Paper 1 title upgrade:
- **Old**: "Physics-Informed Flow Matching for Universal 3D Molecular Generation"
- **New**: "Boltzmann-Guided Flow Matching: Unifying Generative Modeling, Score Matching, and Boltzmann Sampling"

Paper 1 contribution list (updated):
1. Boltzmann-aligned flow matching objective (3-term loss)
2. Boltzmann consistency theorem (Theorem 8)
3. Universal neural-potential grounding (OMol25 pretraining, 83 elements)
4. Bond-free quantum-native parameterization
5. Post-hoc drift unification (Level-1, kept)

The Paper's narrative spine: "we show how to train a generator whose
output IS the Boltzmann distribution, not merely a biased approximation
of it, using three ingredients: flow matching for density, score matching
for gradients, and energy matching for global shape."

---

*End of method doc. Next: write companion Section 3 v2 (`section3_bgfm.md`)
and Appendix A v4 with Theorem 8 proof.*
