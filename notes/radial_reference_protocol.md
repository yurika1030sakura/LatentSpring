# Analytic-divergence reference, September 9

This is a diagnostic architecture, not a claimed new method. Equivariant
pair-kernel CNFs with analytic divergence were already introduced by
[Koehler, Klein and Noe, ICML 2020](https://proceedings.mlr.press/v119/kohler20a.html).
The purpose is to obtain a tractable exact-trace comparator for stochastic
energy objectives. It may underfit relative to the original message-passing
backbone. Do not promote mathematical tractability into molecular usefulness.

For centered coordinates, the reference uses

`v_i = b_g(t,m) x_i + sum_{j != i} phi_ij(||x_i-x_j||^2,t,m) (x_i-x_j)/(N-1)`.

The pair coefficient network sees symmetric sums and products of learned
atom/charge embeddings and time features; it never sees other coordinates.
The radial part has 24 Gaussian kernels of squared distance, centers from 0
to 36 Angstrom squared and fixed width 2 Angstrom squared. Pair coefficients
and global dilation are bounded by 4 using tanh. The one-atom case uses the
zero-dimensional COM subspace. Pair velocities sum to zero and the exact
intrinsic divergence is

`3(N-1)b_g + sum_{i<j} 2 [3 phi_ij + 2 r_ij^2 d phi_ij/d(r_ij^2)]/(N-1)`.

Here phi in the formula excludes the normalizer; the implementation includes
that normalizer inside phi. Translation directions have zero derivative, so
the projected ambient trace equals the intrinsic trace. Smooth squared-distance
kernels avoid dividing by pair distance. For each fixed composition the field
has bounded spatial derivatives, but this does not guarantee a small numerical
error at a chosen step size. Independent solver checks remain mandatory.

The runtime patch attaches new modules to the existing CTMCVectorField instance,
retains and freezes its original parameters, and exposes only the conditional
displacement head. Original weights do not initialize the radial network. Its
composition prior remains the separately frozen original generator. A radial
checkpoint must be patched before strict state loading, and its protocol records
the backbone; it cannot silently be scored as a FlowMol position checkpoint.

## First bounded experiment

- OMol25 remains the training corpus; bond supervision is zero, max_atoms=200.
- Linear displacement FM path to T=1, 10,000 distinct examples, seed 9004,
  AdamW learning rate 0.001, no force or energy term. This is a reference
  trained from scratch; it is not a matched warm-start comparison.
- The same eight development compositions and four fixed Gaussian priors as
  earlier position checks, with both sampling convergence and xTB failures.
- Retain all eight local perturbation groups, with midpoint 16/32/64 steps,
  fixed stochastic probes and the analytic exact divergence on each group.
- Run the small reference on allocated CPU nodes; record actual runtime. It
  does not consume a new GPU while the existing solver checks run.

Analytic divergence values and their parameter/coordinate gradients are checked
against explicit autodiff traces, including coincident atoms and one-atom
graphs. Rotation, translation, permutation, strict checkpoint restoration,
full density differentiation and the discrete adjoint have separate checks.
The analytic trace removes only trace-estimation error, not ODE discretization
error. Its finite-step log-density is still an approximation to the CNF.

## Recorded first result and bounded follow-up

The 10,000-update reference completes in 156.4 CPU training seconds. The local
analytic-trace midpoint-32 to midpoint-64 differences are at most 0.00248 nats,
and the eight-replica mean differences are at most 0.02581 nats. Sampling RMS
drift is at most 0.000625 Angstrom. However, xTB converges for only 16/32 samples,
with a successful-only strain median of 38.23 eV. This is a poor molecular
generator even though it is a useful tractable density control. All failures
are retained in `radial_reference_development_v1` (job 45601575).

Two bounded follow-ups are distinct, not seed replication:

1. Restart the same scratch initialization and data order for 100,000 FM-only
   updates. This tests training scale before attributing the entire quality
   gap to architectural capacity. It remains a scratch model, unlike the
   pretrained original backbone, and needs independent numerical checks.
2. From the 10,000-update reference, run eight 20-update engineering controls:
   FM, exact value/shuffled/zero labels, squared/product with independent/common
   Gaussian probes. All use seed 9012, identical FM batches, learning rate
   0.0002 and, where present, energy weight 0.001, eight parents, two local
   siblings and midpoint-64. Stochastic arms use two independent replicas.
   Record applied/skipped terms, gradient norms and wall time. This is a
   runtime and estimator check on a poor generator, not a molecular result.
   Exact traces have different computational cost and are not advertised as
   equal-cost stochastic estimators. Replica means versus products have equal
   trace counts; common Gaussian and independent controls are kept explicit.
