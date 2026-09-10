# Species coupling first execution protocol

Full goal remains open. This implements the conditioner/group wrappers specified
in species_coupling_adapter_candidate.md, using the centered nonlinear convex
primitive. No model/density approximation gap is introduced by the adapter:
the FlowMol base remains frozen, and the per-sample log volume is explicit.

Conditioner: two invariant message-passing layers, hidden32, radial24, element
and active-role embeddings, global charge/spin/log-target-kT and layer-mode input.
Zero final heads initialize the exact identity. Shared context vectors supply
convex directions and a bounded .1-A centroid shift. No dropout or batch coupling.

One sweep consists of within-element-group updates (when count>1), followed by
adjacent element-centroid pairs in ascending atomic number. Current N8 condition
has7 coupling layers. Context excludes the active internal displacement or active
centroid difference. Every context dependency remains in the differentiable graph.
The within-group log volume uses the COM Schur complement; centroid volume uses
the full3-by-3 derivative. Inverse calls are reconstruction-only, not likelihood
gradient APIs. Small/passive-free contexts limit expressivity; no universality claim.

Seven tests cover full intrinsic Jacobians at nonzero parameters, context
invariance, exact inverse/volume cancellation, O(3), joint coordinate/element
permutation, independent batch rows, identity initialization, complete parameter
gradients and a200-atom complete model. These precede molecular execution.

Molecular screen: same4096 training source samples and256 independent development
samples as the linear/scalar baselines. Train200 AdamW steps, batch16, clip10.
Initialization seed9161; sample-selection seed9141 matches the first linear controls.
Use a predeclared neural learning-rate cosine schedule .001 to .00001, rather than
the linear control's .01 to .0001; different parameterizations have different
sensible scales. Report this distinction and all compute, not only oracle calls.
Budget3200 training+512 assessment queries=3712, with inherited source costs additional.

First run a2-update16-evaluation engineering smoke (64 queries). Then the200-step
screen only if source-energy replay, finite gradients, trained inverse replay and
full trained intrinsic Jacobians pass. After fitting, verify inverse on up to32
heldout rows and full intrinsic determinants on two rows. Evaluate physical energy
on every heldout sample, exact paired endpoint-KL change, inherited valid path work,
and separate xTB/geometry checks. Keep failures and all parent IDs. A negative KL
change does not establish full calibration, effective importance sampling or AI
novelty. Fresh confirmation/training replicas and broader conditions remain required.

This is an architecture candidate. Equivariant coupling/convex/residual flows and
normalizing-flow energy objectives are direct prior art, not claimed new identities.


## Fixed larger-budget comparison after the200-step screen

The200-step model passes trained inverse and full intrinsic determinant checks.
Its development DeltaKL is-.18129+/-.05689 SEM; paired difference versus typed
linear is-.02358+/-.02579, so no clear nonlinear advantage is established.
A best-COM-linear approximation fitted to256 training rows leaves about39% of
the adapter displacement norm on256 separate rows; the model is not exactly
linear, but its total displacement RMS is only .00230 A. This is diagnostic,
not a calibrated-performance result.

Run a fixed1000-update comparison: nonlinear seed9161/selection9141 and typed
linear selection9141. Both use batch16, the same4096 source-training rows and
256 development rows, and16512 oracle queries each. Start from identity, not a
claimed continuation of the shorter cosine schedule. Keep each architecture's
already stated learning-rate endpoints, with cosine decay across1000 updates.
Record all compute and the different parameterizations. No hyperparameter sweep
or best-checkpoint selection. If the nonlinear model does not improve over the
matched typed control, do not extend the same recipe merely because training
loss declines. Global mode-weight learning remains an unresolved possibility,
not a failure mechanism already proved by the current data.


## Fresh comparison after the1000-update run

The nonlinear1000-step development DeltaKL is-.23491+/-.06397 SEM, versus
-.16686+/-.05381 for matched typed linear. Their paired difference is
-.06805+/-.03022, a small nominal advantage requiring confirmation. Freeze both
checkpoints; generate2048 new base samples with seed9169 and evaluate both maps
on the identical rows. Total confirmation cost6144 oracle calls (base and two
maps). Verify fresh nonlinear inverse and volume cancellation; keep all work
weights and failures. This is one-condition/one-training-seed confirmation only.
The original source reverse model is used for fresh weights, so do not pool
with the development auxiliary's ESS. No parameter or selection changes follow
from this panel until all outcomes have been recorded.


## First confirmation and training replica

The fresh2048-sample evaluation gives nonlinear DeltaKL=-.17451+/-.02808 SEM,
linear=-.12558+/-.01829, and paired nonlinear-minus-linear=-.04892+/-.01732.
Importance ESS remains approximately1/2048 for both, so no calibration or broad
sampling efficiency is claimed. The difference needs training replication.

Repeat the fixed1000-step recipes with nonlinear initialization9162 and sample
selection9142, and typed-linear sample selection9142. Every other setting and
16512-query budget is unchanged. Evaluate the frozen pair on the same stored
2048 base samples, with common energies/random numbers explicitly acknowledged.
This tests training-seed variation; it is not a new independent evaluation
stream. New transformed-energy queries total4096; base energies are reused.
Retain both seeds regardless of outcome. Broader conditions and a stronger
coupling-flow baseline remain required before any ICLR contribution claim.
