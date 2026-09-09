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
