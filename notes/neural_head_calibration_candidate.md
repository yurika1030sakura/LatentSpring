# Frozen learned-feature calibration candidate

Previous goal turn classification: progress. The four-direction calibration
implemented a valid component and reduced independent estimated score risk, but
left large unfitted radial/angular errors. The full ICLR objective remains open.

## Intervention

Freeze each antithetic500-step critic (seeds9101 and9103). Extract the32 bounded,
invariant features immediately before its final linear energy head. Together
with the quadratic scale potential, their gradients define33 conservative score
features. The original score must reconstruct from the head coefficients and
these gradients; require max discrepancy<1e-7 and preserve the checkpoint state.
Refit residual coefficients by the existing constrained ridge solve, .01 after
feature-RMS scaling, retaining at least half the positive quadratic tail.
This is a convex readout correction, not further SGD on the failed recipe.

The neural feature map is learned, but neither score matching nor its linear
solve is new. Direct prior art includes Learning Deep Kernels for Exponential
Family Densities (ICML2019), https://proceedings.mlr.press/v97/wenliang19a/wenliang19a.pdf ,
along with Stein gradient estimators and NDSM controls cited in the earlier
notes. Novelty and molecular sampling advantages remain unestablished.

## Finite-noise moments without neural Hessian traces

For y_plus=mu+sigma*epsilon and y_minus=mu-sigma*epsilon, define
 d_j=epsilon dot(f_j(y_plus)-f_j(y_minus))/(2sigma).
Gaussian integration by parts gives E[d_j]=E_q div f_j exactly at the actual
finite sigma. This is not a zero-noise approximation. Use exact divergence d
for the known quadratic-scale feature. Fit G from the average of both feature
outer products and r from average score-feature products plus d_j.
The random-noise pair is one independent unit for uncertainty calculations.
The population projection and tail properties remain those in
notes/stein_calibrated_entropy_candidate.md; estimated moments require validation.

## Fixed development experiment

Calibration:8192 parents, seed9107/noise9108. Development:16384 parents,
seed9117/noise9118. The latter panel has already been inspected for version1,
so this is not blind confirmation. Use both plus/minus samples derived from
recorded means/noise; no forward-model update or energy query. No new basis or
ridge is chosen from development outcomes. Ridge .01, batch64, all rows retained.

Arms: learned features at both seeds; random frozen features of the same critic
architecture, seed9121, with a Gaussian base; and a strong fixed-feature control
using element-pair-specific radial functions (8 centers .75 through4.25 A,
width .5) plus scale. The typed control has105 features for this condition,
more than the33 learned features; report compute and size rather than claiming
parameter matching. The base Gaussian precision is fixed from previous training
moments. Keep all four arms.

Assessment: the same14 scale/radial/size/angle moments, relative score-risk change
using paired finite-noise Stein estimates, and paired DSM against both baselines.
Scale belongs to the fit; radial tests can correlate with fixed radial features.
Do not treat finite moment checks as global qualification. A development gate
requires both learned seeds to show risk improvement, pass all14 moments and
beat both baselines by2 SEM in paired DSM. Only then generate a new independent
confirmation panel before any actor pilot. Otherwise stop this precise recipe.

Run a32-parent engineering subset before the full panels; it cannot qualify
scientifically. Bound each run to0.5 GPU-hours. Save all hashes, coefficients,
condition numbers, active constraints, paired rows and failures. Checkpoint
reconstruction and feature symmetry tests must pass first.
