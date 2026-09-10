# Candidate: residual-calibrated endpoint-entropy learning

This is a concrete candidate method designed for the measured score-estimation
failure. It is not yet validated molecular sampling or established AI novelty.
It combines a learned proposal score, a conservative finite-dimensional residual
calibration, and an endpoint-gradient update using the physical target force.
This experiment implements and tests the calibration layer before any actor
update. Earlier failed critics and all negative results remain intact.

## Population property and its limits

Let q be the actual generator's endpoint density and s0 an approximate score.
Let f_j=grad psi_j be conservative test fields with valid Stein integration by
parts, and let F c=sum_j c_j f_j. Define
G_ij=E_q[f_i dot f_j], r_j=E_q[s0 dot f_j+div f_j].
Then r_j=E_q[(s0-s_q) dot f_j]. The corrected score s_c=s0-Fc has risk difference
E_q||s_c-s_q||^2-E_q||s0-s_q||^2=c^T G c-2 c^T r.
Thus ordinary quadratic projection removes the component of score error in the
chosen span. Ridge fitting minimizes .5 c^T G c-c^T r+.5 lambda||D c||^2,
where D contains per-feature RMS scales. Keep zero correction feasible while
requiring c0>=-kappa0/2 for psi0=||x||^2/2 and base tail coefficient kappa0>0.
This retains positive quadratic confinement. At population moments the minimum
has nonpositive objective and hence cannot increase score risk. This is a
standard score-matching/projection identity, not a claimed new theorem.

Estimated moments do not inherit that guarantee automatically. For coefficients
fitted independently of assessment samples, the per-sample quantity
||F c||^2-2 sum_j c_j(s0 dot f_j+div f_j)
is an unbiased estimate of the expected Fisher-risk change under the same Stein
conditions. We report its sampling uncertainty. This is a relative risk check,
not a certificate that the entire score or generator distribution is correct.
A smaller score risk does not by itself guarantee a finite actor step improves
Boltzmann sampling. No such claim is made before an actual actor comparison.

The scalar corrected potential is E0+sum c_j psi_j. Bounded radial potentials
and the positive quadratic tail ensure it remains normalizable on COM-free H.
Rotation/translation/permutation symmetries are preserved. The density of this
critic is not substituted for the actual generator density.

## Complete learning-loop specification

For each declared composition/charge/spin and each generator update, freeze the
current actual Gaussian-path generator q_theta. Use separate samples for neural
critic fitting, residual calibration and qualification; an update minibatch is
also independent of those fits. Train the critic using actual terminal-noise
pairs, obtain the constrained calibration, then use the corrected score in
endpoint_kl_surrogate with target score (force_eSEN-kappa*x)/kT in the same COM
basis. Hold critic and calibration parameters fixed during the actor gradient.
After changing the generator, its proposal score must be refreshed and checked;
a stale critic is not accepted as the score of the new q. Final assessment uses
independent generation and the actual path weights, with geometry and compute
reported separately. The conditional composition prior is not changed.

Only the frozen calibration component is implemented and evaluated in this
experiment. The molecular actor loop has not been run or qualified. A projected
risk decrease tightens a Cauchy-Schwarz upper bound on actor-gradient bias when
the generator Jacobian has a finite second moment; it does not imply that each
realized finite actor update decreases the true KL. This standard inequality is
not an additional novelty claim. The full method must earn its claim through
actual controlled generation experiments.

## Direct prior art

Stein gradient estimation and score matching from implicit samples are established:
Li and Turner, ICLR2018, https://arxiv.org/abs/1705.07107 ; Shi et al., ICML2018,
https://proceedings.mlr.press/v80/shi18a.html . VSD/DMD and NDSM controls are
already recorded in notes/endpoint_entropy_candidate.md. Residual projection,
constrained ridge regression and a finite set of Stein moments are not enough
novelty on their own. Any paper contribution needs a distinct effective learning
intervention and replicated total-compute benefits beyond those methods.

## Fixed first molecular experiment

Base scores: the two antithetic500-update critics, seeds9101 and9103. Keep both,
not the better one alone. Fit calibration to the existing8192-parent development
panel (parent seed9107, final noise9108), which did not train either neural
critic. Calibration features: scale and radial Gaussian potentials centered at
1,2,3 A, width .5, averaged over unordered pairs. Ridge .01 after feature-RMS
scaling. Enforce the positive-tail constraint as above; no hyperparameter sweep.
Include a Gaussian-base-plus-identical-calibration ablation, so success cannot
be credited to the neural network without beating the simple calibrated model.

Generate a new16384-parent panel, seed9117, final noise9118. It is independent
of neural training and calibration; there are no new energy queries. Use all
rows. Assess the original four fitted moments, but never count those alone as
validation. Additional unseen checks: radial centers .75,1.25,1.75,2.25,2.75,3.25
with width .35; radial size scales3,5; and two symmetric triangle-angle probes
with envelope scales2,3. The angular divergences are computed exactly only for
this small eight-atom diagnostic, with independent Hessian tests.

Report heldout Fisher-risk differences from the Stein formula, paired denoising
risk differences using the actual final noise, every fitted/unseen Stein moment,
all coefficients, condition numbers, active constraints, hashes and costs.
A necessary gate requires both corrected neural seeds to improve relative risk
and beat the calibrated Gaussian in paired DSM by2 SEM, and to place every
fitted and unseen moment within3 SEM. Even a pass permits only a bounded actor
experiment, not an ICLR/calibration claim. Failures remain in the denominator.


## Completed version1 result

Fresh-panel job45820074, engineering smoke45820248 and full audit45820530 all
completed.253 tests pass. On16384 new samples, both calibrated neural seeds
show negative heldout Stein estimates of Fisher-risk change per coordinate:
-.15672+/-.03340 and-.89205+/-.07398 (mean+/-SEM). Paired denoising-risk estimates
are noisier: -.01314+/-.18617 and-1.63841+/-.46858. The two estimators concern
the same expected risk difference; neither is a global score certificate.
Both neural versions beat the identically calibrated Gaussian baseline in DSM.

All4 fitted moments pass for both seeds, but only2/10 and4/10 unfitted moments
pass. Remaining violations reach20.72 and13.23 SEM, including angle probes.
Therefore no molecular actor update is released. The correction improves a
measurable component of score error, but the fixed four-direction version is
insufficient. Full result: research/evidence/score_calibration_audit_v1.json.
No energy-oracle query or forward-model update was needed for this experiment.

A prospective next variant would calibrate in learned invariant feature
directions, rather than the fixed four directions. That variant is not yet
implemented or evaluated. In particular, refitting the linear head of a frozen
invariant energy network is a concrete way to test representation versus
optimization error through a richer convex score-matching problem. Its prior
art must be checked and its validation directions kept separate from fitting.
Do not append the failed assessment probes to the fit and then call their
training fit a validation success. The16384 panel is an independent result for
version1; after using its outcomes to design a new variant, it becomes development
data for that variant, which needs new independent confirmation.
