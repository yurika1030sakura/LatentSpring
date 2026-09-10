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
