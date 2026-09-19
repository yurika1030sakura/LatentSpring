# Frozen-parent physical connection head

Broader physical targets improve held-out joint quality over matched
original-reference fine-tuning by5.66 percentage points, but their1.37-point
gain over the unadapted parent is uncertain and differs in sign across seeds.
The next candidate learns a local geometry residual while preserving every
pretrained parent tensor. This tests a concrete alternative to updating the
entire generator; reduced forgetting is a motivation, not an established result.

## Architecture

The parent remains the harmonic-source, two-pass FlowMol generator. From its
current velocity estimate v0 at coordinates X and flow time t, form the provisional
endpoint H=X+(1-t)v0. The correction head sees only X,H,atomic identities,t and
atom count. It receives neither supplied bonds nor the true training endpoint.

A symmetric pair MLP takes sums/products of atom embeddings, scaled distances
in X and H, alignment of the two relative vectors, time features and atom count.
Its two coefficients are bounded by tanh. Geometric contact weights decrease
smoothly with the parent's predicted distance relative to covalent radii. They
are not calibrated chemical-bond probabilities.

For an unordered pair, combine the two normalized relative vectors, multiply by
the contact weight, and add opposite messages to its endpoints. Normalize all
messages within a molecule by twice its maximum weighted degree (floored at one),
then apply the t^2 gate and a1-Angstrom/unit-flow-time scale.

The construction yields zero total correction, rotation/permutation equivariance,
and per-atom instantaneous correction norm at most t^2. These statements follow
from pair symmetry, antisymmetry, bounded coefficients and the degree bound.
They do not imply a bound on final displacement after interaction with the parent
ODE, preserved chemical connectivity, or exact target-distribution recovery.

The final MLP layer is initialized to zero, giving exactly the parent field at
initialization. The head has8178 trainable parameters with the production83-element
vocabulary. Parent tensors, including buffers, are verified unchanged after
head fitting. The supervised objective uses the corrected final velocity; the
frozen parent's first-pass loss is constant and omitted.

Equivariant pair messages are established work; see
[E(n) Equivariant Graph Neural Networks](https://proceedings.mlr.press/v139/satorras21a.html).
The candidate contribution is this task-specific physical residual and its
measured utility on a frozen composition-only flow, not a new equivariance law.

## Frozen comparisons

`research/evidence/physical_connection_v1.json` specifies:

- The same128-composition empirical reference/physical target bank, reused without
  any new teacher queries.
- Two stochastic training seeds, each with2000 updates per method.
- Physical-target full fine-tuning, original-reference correction-head training,
  physical-target correction-head training, and the unadapted parent.
- The same row choices and augmentation/source noise within each seed.
- A separate16-composition panel not used by the completed broad study or any
  earlier generator panel, with16 raw draws per composition.

Every inference model includes the correction head. Parent and full-FT controls
receive a zero-output head with exactly unchanged parent tensors. All therefore
use128 backbone calls plus64 small-head calls per attempted output. Do not label
this as128 total neural-module calls. Full FT and head fitting both use4000
backbone forward examples, while head fitting additionally uses2000 small-head
forward examples and only backpropagates through the head. Timings and counts
are reported separately; there is no equal-wall-time claim.

The primary contrast is the physical correction head versus its unadapted parent
under all-attempt GFN2 graph-valid force<=5 yield. Original-reference correction
and full fine-tuning are required controls. All seeds and thresholds are retained.
This study does not establish global Boltzmann sampling or superiority to GAGA.

Tests exercise identity initialization, rotation and permutation behavior,
centering, correction bounds, coincident atoms, real FlowMol gradient flow,
unchanged parent parameters, and the sampler's actual correction-call count.
Together with the existing source/adapter tests,9 tests pass in the real training
environment. Trained generation results remain to be established.

## Completed result

Array47245883 and audit47245885 completed. The physical-target head does not
improve the parent: joint GFN2 force<=5 yield is32.62% versus33.40%, a difference
of-0.78pp [-1.76,0.20]. It equals the reference-target head's pooled yield.
The comparison with full fine-tuning is+3.71pp [-0.59,8.20]. All parent tensors
remain unchanged, but that architectural property is not a quality improvement.
Median coordinate change versus the parent is0.0085 and0.0053 Angstrom in the
two runs. Insufficient correction is a possible motivation for changing training
states, not a proved explanation. The next bounded candidate uses cached states
from actual parent trajectories and local force/work targets. No failed branch
is adopted into the manuscript's method. Registry v24 includes all2048 new
evaluation outputs,12000 optimizer steps and2048 GFN2 attempts.
