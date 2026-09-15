# Current harmonic-source paper and audited controls

Read `research/HARMONIC_PAPER_STATE_20260915.json`,
`notes/source_sc_covariance_results_20260915.md`,
`notes/harmonic_novelty_and_physics_20260915.md` and
`notes/native_endpoint_results_20260915.md`.
The complete internal manuscript is `paper/tree_working.tex` / `.pdf`:
6 main pages,11 total, clean compile and no unresolved references or overfull boxes.

The harmonic source beats isotropic Gaussian by5.86pp on12 development
compositions and8.83pp on10 additional compositions using frozen models.
It also beats the calibrated covariance Gaussian by11.65pp and15.39pp;
both continuation seeds favor harmonic in both panels. Covariance estimation
uses128 trees and has1.85–6.25% Frobenius error against8192-tree calibration.
This supports a source-structure benefit beyond the tested covariance
approximation; exact-moment or universal causal attribution is not established.

The original30k endpoint checkpoint was also evaluated with two sampling streams,
original endpoint semantics,127 Euler steps and128 actual denoiser calls. Counts
are106/113 with native predicted history and116/120 with fixed-category history
and centered bootstrap, versus the adapted harmonic413/375, each out of640.
This is task adaptation with different training histories, not a published native
benchmark reproduction or an equal-total-training-cost source ablation.
The fixed-category variant changes bootstrap centering as well as categories.

All jobs are terminal:46504779 covariance and46506935 native, as well as previous
46497703/46503573. No new experiment is queued. Registryv10 includes all outputs.
Keep the selected harmonic models frozen. No architecture or hyperparameter sweep.

AI contribution claim: explicit random-tree harmonic source and its integration
into composition-conditioned, bond-free molecular FM. Random-tree priors,
self-conditioning and Jarzynski are established. Distinctive task-specific utility
now has controlled support; sufficient ICLR originality is not yet established.
Physics is the auxiliary spring prior and atomic scales. Jarzynski remains in
an assumptions-explicit appendix; no work weights, Boltzmann law or output ESS
were measured for current samples.

Next essential evidence concerns competitiveness with independently trained
molecular generators and energy/geometry quality under the same task, plus broader
robustness. Do not claim SOTA or demand every composition win. Research goal remains
unachieved; the draft is not scientifically submission ready. Reserved722 outcomes
remain unqueried. The user handles authors and submission.
