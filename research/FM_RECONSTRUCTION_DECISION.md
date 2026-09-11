# FM reconstruction decision

The user authorizes restructuring the FM architecture if it improves this
project. The current system remains FM-based: frozen conditional FM generates
the source; a separately trained invertible transport applies energy/entropy
refinement. The refinement objective is not the original FM loss.

We should first repair/compare the parts with measured limitations. Fixed-index
splitting does not preserve same-element permutation equivariance and cannot
be the claimed homogeneous repair. A replacement must retain permutation and
O(3) equivariance, change meaningful collective coordinates, admit a valid
inverse and account for the full intrinsic Jacobian. A global equivariant
pair-interaction convex map is one candidate for a bounded prototype; it would
need full constrained-volume accounting and a fresh complexity/expressivity
audit. It is now implemented and passes constrained-Jacobian, inverse and symmetry
checks. Two seeded pilots improve relative KL but retain very low ESS; no
sampling-efficiency or novelty claim is established. See
`evidence/pair_capacity_decision_v1.json`.

For the molecular task, finish the frozen48-arm campaign and strong matched-source
controls before attributing failure to the FM backbone. Poor finite-sample overlap
is not proof that the base has zero support; its positive terminal Gaussian gives
full support on the stated COM subspace. The old path ESS failure also does not
isolate base quality from the chosen path/reverse construction. Qualified
independent references are needed to identify missing useful probability mass.

If those comparisons identify the FM representation or learned source distribution
as the limiting component, rebuild that component on a separate versioned branch:
state/electronic conditioning, intrinsic geometry, symmetry and target-aware
training must be explicit. OMol examples cannot simply be relabelled as a300-K
Boltzmann ensemble. Do not make a temperature embedding alone a calibration claim.

Unfreezing the FM base changes its entropy. The current unknown-source-entropy
cancellation would then be insufficient for end-to-end energy training. A new
valid density, entropy-gradient or path estimator and corresponding independent
checks would be required. Loss/architecture reconstruction remains authorized;
replacing the entire backbone solely to manufacture novelty is not evidence-based design.

The next comparison is enabled by the now-qualified JAX/EACF energy bridge.
A fixed-source EACF joint-entropy adapter and a standalone EACF/FAB model have
different source and objective contracts. Preserve that distinction, physical
query counts, auxiliary-gap limitations and independent geometry/coverage metrics.
Do not shrink the ICLR objective to a working interface or a positive toy.
