# Non-equilibrium work as a BGFM foundation

Decision, 2026-09-09: adopt the valid identities as a theoretical foundation and
implement a separate correction/teacher branch. Do not present the identities,
AIS, stochastic path weights, or their combination with learned transport as new.
Do not rename historical results as Jarzynski-corrected Boltzmann generation.

## What the AFM connection means

Jarzynski's equality connects work measured on repeated externally driven paths
with an equilibrium free-energy difference, starting from the initial equilibrium
distribution: E[exp(-beta W)] = exp(-beta Delta F). Atomic force microscopy and
optical pulling experiments are applications. The controlled pulling parameter,
the force measurement and the definition of work must agree; an arbitrary
force-line integral along a learned denoising trajectory is not sufficient.

Primary sources:

- Jarzynski (1997), https://arxiv.org/abs/cond-mat/9707325 .
- Hummer and Szabo (2001), free-energy reconstruction from nonequilibrium
  single-molecule pulling, https://pmc.ncbi.nlm.nih.gov/articles/PMC31107/ .
- Vaikuntanathan and Jarzynski (2008), escorted simulations using an artificial
  transport field, https://arxiv.org/abs/0804.3055 .

The closest verified Microsoft paper is Xie et al., Enhanced Diffusion Sampling
(2026), https://arxiv.org/abs/2602.16634 . Its Section 2 explicitly takes the
pretrained model distribution as the unbiased equilibrium reference. Steering
and unbiasing recover that reference; this does not independently prove that our
FM model is the Boltzmann distribution of the eSEN potential. It builds on
Feynman--Kac correctors: https://arxiv.org/abs/2503.02819 . We have not established
which particular Microsoft paper the user remembers.

FEAT, https://arxiv.org/abs/2504.11516 (NeurIPS 2025), is even more directly
relevant to the numerical repair: Section 3.4 uses forward/backward discretized
Gaussian transition probabilities to compute corrected generalized work without
divergence integration. SNF, https://arxiv.org/abs/2002.06707 (2020), already
combines stochastic blocks, learned transformations and exact path weights.
These are required comparators, not missing prior work that BGFM can claim.

## Fixed-condition target and the existing residual

Fix atom composition, total charge and spin multiplicity c. Work in an
orthonormal basis of H={x:sum_i x_i=0}, dimension d=3(N-1). A density is relative
to the induced Lebesgue measure. Specify a restrained/domain-limited potential
U_c and assume 0<Z_c=integral_H exp(-beta U_c)dx<infinity. Removing translation
alone does not confine dissociated atoms. A restraint changes the target and
must appear in both the model description and evaluation. No statement here
determines mixture weights between different c.

For a normalized prior q0 and a smooth invertible map x1=Phi(x0), define

    w(x0) = beta U_c(x1) + log q0(x0) - log |det D Phi(x0)|
          = beta U_c(x1) + log q_theta(x1).

Change of variables gives E_q0[exp(-w) f(x1)] = integral exp(-beta U_c(x))f(x)dx.
In particular E exp(-w)=Z_c and E_qtheta[w]+log Z_c=KL(q_theta || pi_c).
Thus the ideal BGFM density-energy residual is a dimensionless generalized
work. A CNF integral gives log|det D Phi| only for the corresponding exact flow,
not automatically the discrete numerical map. These are standard identities.

The old within-parent off-policy variance is not the Jarzynski ensemble average.
It removes each cloud's offset and leaves cross-cloud basin mass unidentified.
Low variance on a few perturbed structures cannot certify low dissipated work
on all generated paths, global coverage, or calibrated generation.

## Finite-step correction that avoids a trace estimate

Let Q(x_0:K)=q0(x0) product_k K_k(x_k|x_{k-1}) be the actual implemented forward
path, and choose normalized auxiliary backward kernels L_k(x_{k-1}|x_k).
If Q covers the support of the target path measure, use

    log a = -beta U_c(xK) - log q0(x0)
            + sum_k [log L_k(x_{k-1}|x_k) - log K_k(x_k|x_{k-1})].

Multiplication by Q cancels q0 and every forward kernel. Integrating the backward
kernels one at a time leaves exp(-beta U_c(xK)); hence E_Q[a f(xK)] is the
unnormalized target integral. This proof works at the chosen finite time step.
It needs neither a learned marginal likelihood nor a claim that L is the true
physical reverse dynamics. Forward/backward Gaussian densities must correspond
to the state actually sampled; projections, rejection, clipping of positions,
or unrecorded guidance change that density. The public API therefore operates
on Euclidean coordinates and supplies an orthonormal molecular COM basis.

The implementation accumulates float64 log weights, guards invalid oracles, and
reports ESS, maximum normalized weight and log mean weight. The mean of raw
weights is an unbiased Z estimator under the assumptions. Its logarithm and
self-normalized expectations have finite-sample bias. Resampling does not turn
a finite weighted cloud into independent exact target samples. Very low ESS
can make a formally valid method practically useless, especially in path space.

For a value-only reference, implement Neal's AIS
(https://arxiv.org/abs/physics/9803008):
gamma_b=q0^(1-b) exp(-b beta U), weight before each transition by the change in
bridge energy, then use a random-walk Metropolis kernel invariant to gamma_b.
An exact invariant kernel need not mix to equilibrium at every bridge. The
current implementation requires finite log density values at all query points;
hard-wall targets need explicit support-aware logic before use.

## Why the previous trace-noise correction does not solve work weighting

If log q_hat=log q+epsilon with E[epsilon|x]=0, generally
E[exp(-epsilon)|x] != 1. For conditional Gaussian noise with variance s(x)^2,
the factor is exp(s(x)^2/2). Variable trace noise therefore changes target basin
weights even with arbitrarily many sampled paths. For two states with target
masses .2/.8 and log-noise variances 0/2, the limiting right mass becomes
(.8 exp(1))/(.2+.8 exp(1))=0.915776..., rather than .8.

Independent replica products remove a quadratic-estimator bias, not this
exponential bias. Never insert the existing noisy CNF readout into work weights
and claim exact correction. Nor is terminal exp(-beta U) alone sufficient for
samples from a nonuniform proposal; it targets proposal(x)*exp(-beta U(x)).

## Training and evidence gates

Use globally weighted paths within each fixed c to supply a teacher cloud.
Pair each endpoint with independent Gaussian noise and a linear interpolation
time, then minimize its weighted conditional flow-matching velocity error.
This preserves flow matching and bond-free supervision. Teacher weights are
detached and normalized within c; the resulting FM student is approximate and
must be evaluated separately from the weighted teacher.

`cfm_mol/nonequilibrium.py` implements the work factors, Gaussian path sampler,
AIS reference, molecular basis and weighted-CFM loss. `tests/test_nonequilibrium.py`
checks exhaustive finite-state endpoint masses with arbitrary non-reversible
kernels, an exactly reversible Gaussian step with constant work, Gaussian AIS
moments/normalizer, energy-offset invariance, COM measure and training gradients.
The Gaussian regression test initially included an unsupported ESS>0.6 threshold;
the observed ESS was 0.5402 while all target checks passed. It now asserts the
definitional ESS range, not invented algorithmic efficiency.

`scripts/research/toy_nonequilibrium.py` runs five independent seeds on a known
2D Gaussian mixture, retaining unweighted, energy-only and AIS corrections and
matched weighted/unweighted FM distillation. Report both basin mass and within-
basin spread: energy-only weighting can accidentally get one mass right while
narrowing the modes incorrectly. Toy success reproduces known mathematics;
it establishes an implementation baseline, not novelty or molecular benefit.

Before promotion to the main paper: compare matched molecular FM-only, AIS,
learned-path correction and stronger existing methods at equal oracle cost;
retain failed energies and poor-ESS conditions; measure independent physical
quality, mode coverage, sample diversity and student error on preserved-metadata
held-out systems. Current OMol val/test duplication still blocks a blind-test
claim. The eSEN checkpoint has been found and CPU-load verified; raw source IDs,
unclipped electronic-state metadata and an independent test set remain needed.

ICLR judgment: this supplies a sound repair path and a sharper experimental
question. It does not by itself add a novel contribution or establish an
acceptable paper. A demonstrated transferable improvement per oracle/compute
budget, beyond these prior methods, is still the critical missing evidence.
