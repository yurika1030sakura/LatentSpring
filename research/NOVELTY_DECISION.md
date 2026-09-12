# Current contribution decision — September 12, 2026

There is an implemented, independently audited framework for reversible joint
molecular connectivity and geometry proposals. There is not yet an established
competitive learned contribution. The constrained arc decoder is the newest
foundation; `research/GEODESIC_RECONSTRUCTION_STATE_20260912.json` records its
exact evidence. The old normalized-site learner should not be scaled again.

The intended contribution is useful learning of conditional geometric proposals
for molecular connectivity changes, with exact computable correction and measured
benefits under realistic costs. No new physical law is required. MH, vMF fitting,
force/work labels, masking, equivariance, generic geodesic walks and decoder-order
mixtures are established ingredients. A geometric validity repair alone is not
AI novelty. See `notes/support_constrained_geodesic_proposal.md` for the new
construction and geodesic prior art, and `notes/normalized_site_novelty_review.md`
for the previous learner's nearest methods.

The completed strong concentration-64/400 controls remove the original learner's
demonstrated benefit. Its old concentration-10 GFN2 endpoint check cannot qualify
new physical endpoints, and frozen transfer has no average demonstrated advantage.
All negative controls and initialization failures remain part of the evidence.

The new local force/work surrogate is accurate near its fit points but fails as
an unrestricted probability teacher. Constraining its support fixes geometric
validity; it still yields less expected single-step energy decrease than uniform
arcs, despite higher acceptance. Do not equate fit, normalization or acceptance
with useful exploration. The next neural representation must be tested on actual
nonlocal work and complete-chain progress, not only local teacher imitation.

Producer46176878 and audit46176906 are complete on48 fixed FIT parents.
At128 calls, physical site arcs improve mean potential change by0.07615 eV over
legacy site64, with22.4% greater measured sampling runtime. All24 arms and4,668
joint MH ratios are verified. Uniform arcs show no established full-chain gain.
These outcomes guide design; no new learned model or blind benchmark is involved. The evaluated six-composition set cannot fit weights,
and722 reserved outcomes remain untouched. A final scoped claim must include
representative learned-generator/MCMC baselines as well as strong physics,
held-out compositions, full costs, ablations and relevant distributional checks.

Terminal unlike H/halogen exchanges still limit learned connectivity coverage.
The broader audited physical fragment maps exist but have no trained fragment
proposal model; do not imply their success is evidence for singleton learning.
Do not infer universal coverage from OMol25 or from a locally accepted move.

The target itself must have support. The pinned RDKit connected validator rules
out training condition6 by a necessary degree bound; retain its zero denominator
and diagnose it as target incompatibility. The other zero condition remains
unresolved. A future scope precheck must precede outcomes and is not sufficient
to prove positive support.

The goal remains active and scientifically unachieved. Read
`research/CLAIM_AND_BENCHMARK_SCOPE.md` for the user's standard; perfection is not
required, but correct and useful evidence for the claimed contribution is.
Previous decision: `notes/archive/novelty_decision_through_sharp_training_20260912.md`.
