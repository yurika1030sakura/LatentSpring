# Active research status — September 10, 2026 UTC

**Not yet submission ready.** The project now has a specified conditional
sampler, full density gradients and independent generation checks. No repeatable molecular advantage against matched strong baselines has yet
been established; the first independent three-atom calibration is recorded below. The old CNF
density remains numerically unresolved. The finite-path branch avoids that
likelihood integration, but sampling coverage and target suitability remain
unresolved scientific gates.
Authors and submission accounts are outside the user's requested execution.

Current manuscript direction and claim boundaries: `PAPER_STORY_CURRENT.md`.
The existing PDF main text is still an audit/development draft, not a completed
new-method paper.

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-research`. Frozen original audit: `../audits/bgfm_20260908`.
Every submitted job uses a committed source snapshot; `jobs.jsonl` also records
rejected submissions. A scheduler state of COMPLETED alone does not certify
that all scientific checks passed.

## Latest completed candidate: calibration improves partial score risk only

Residual score calibration is implemented and independently evaluated. Fresh
panel45820074 (16384 new parents), smoke45820248 and audit45820530 completed.
The two neural seeds have heldout Stein risk-change estimates -.1567+/-.0334
and-.8920+/-.0740 per coordinate (mean+/-SEM), and both beat an identically
calibrated Gaussian in paired DSM. This is a limited positive score-estimation
result. It is not improved molecular sampling or an established AI contribution.

Both seeds pass4/4 fitted moment checks on new samples, but only2/10 and4/10
unfitted checks. Violations reach20.72 and13.23 SEM. Neither critic qualifies
for an actor update. All coefficients, tail constraints, risks and failures are
preserved in evidence/score_calibration_audit_v1.json.253 tests pass; no new
oracle query or generator update was used. No current job remains active.
The updated manuscript builds with9 main pages at
runs/verification/paper_20260910_calibration/main.pdf; scientific readiness is false.

The complete candidate learning loop is specified in
notes/stein_calibrated_entropy_candidate.md; only its frozen calibration layer
has been implemented and tested. Stein projection and the population guarantee
are established mathematics. Any ICLR contribution still requires a distinct
learning intervention with replicated full-sampling benefits.

## Earlier residual score calibration protocol

A concrete conservative score-calibration layer is implemented, with a
population projection argument and explicit finite-sample limitations. It uses
existing Stein/score-matching principles, not a claimed new identity. The first
fixed molecular test fits four calibration features on the existing8192-parent
development panel and evaluates16384 new parents (seed9117, noise9118), including
unfitted radial, size and angular probes and a calibrated Gaussian ablation.
Two frozen neural-critic seeds are retained. No molecular actor update occurs.

Five new mathematical/symmetry/Hessian tests pass; the full suite passes253.
Fresh panel45820074 completed without oracle queries; calibration smoke45820248
completed with16 assessment rows and cannot qualify the method. The full audit
is submitted. See notes/stein_calibrated_entropy_candidate.md for the complete
candidate learning loop and the frozen-component experimental boundary.

## Latest completed checkpoint: all score qualifications failed

Both3000-update antithetic and grouped-IID runs completed at seeds9101/9103.
Their maximum absolute Stein residual/SEM values are18.61/12.87 and26.20/31.83;
all exceed3. The fixed longer-training recipes are stopped. The proposed16384
parent promotion panel is not generated because no candidate qualifies.
No molecular actor update has occurred, and no ICLR-level method advantage or
novelty has been established.248 tests pass. The updated paper builds with9 main
pages at runs/verification/paper_20260910_score/main.pdf; formatting passes,
scientific readiness remains false. All jobs in this diagnostic chain
are complete; there is no external blocker.

Complete aggregation and figure: evidence/score_qualification_complete_v1.json,
.pdf and.png. Reproduce with scripts/research/summarize_score_qualification.py.
The updated manuscript development appendix includes the negative global
innovation result, scalar entropy mechanism, initial score-screen pass, failed
independent confirmation and all four longer-training failures. Initial passes
must not replace the failed confirmations in summaries.

## Earlier score confirmation and longer-training protocol

The selected antithetic critic and its independent seed both fail the fresh8192
parent confirmation (45817604/45817720). Seed9101 fails radial3 at4.60 SEM;
seed9103 fails dilation and radial2. This overturns the initial2048-row pass as
a reliable qualification result. All evidence is preserved and no actor update
has occurred.248 tests pass.

A fixed3000-step comparison will test optimization sufficiency with the same
architecture: antithetic and grouped-IID at seeds9101 and9103, same per-update
score-point budget, no energy queries. Existing8192 rows are development data;
any later candidate requires a new16384-parent confirmation before promotion.
This remains standard score-estimation research with no validated AI novelty.

## Earlier score-critic screen and independent confirmation

All500-update frozen-critic controls completed without energy queries or forward
updates. Only antithetic noise passes the first necessary2048-parent gate:
paired DSM improves over zero/Gaussian controls and all four Stein moments are
within3 SEM. Raw, scaled IID, grouped IID and mean-control-variate alternatives
fail at least one Stein moment. All arms and gradient-scaling/clipping limits
are retained in evidence/proposal_score_controls_v1.json. This is an auxiliary
score result on a frozen generator, not improved molecular sampling or novelty.

Fresh8192-parent confirmation45817604 is submitted, seed9107 and final-noise
seed9108. An independent critic seed9103 will use the same new parent/noise panel
only after the first job completes. The gate remains unchanged. Both must pass
before any molecular actor update. None has been made. All previous failures,
including the global Gaussian innovation recipe and empirical CFM students,
remain part of the evidence.

## Earlier frozen proposal score protocol

The global innovation diagnostic45815623 completed in5m11s (32 oracle queries,
plus4 smoke queries). Sequential ESS1.915/32; global metric0/.01/.1/1 ESS1.012/
1.001/1/1. No global arm reaches4/32 or improves the sequential control, so this
four-iteration Gaussian recipe is stopped. Its posterior mode residuals remain
large for some paths; this does not rule out converged or non-Gaussian inference.
Coordinate/probability/gradient checks pass and forward weights stay unchanged.
Evidence: evidence/innovation_posterior_5846_v1.json.

A prospective endpoint-entropy update is implemented with direct VSD/DMD prior
art and no novelty claim. In the known scalar mechanism check, five fitted-critic
seeds reach variance .24818--.25142 versus target .25; fixed-independent reverse
and stale-critic controls collapse to .04. This is not a molecular result.
The necessary next gate is an independently tested frozen molecular proposal
score. A small invariant energy critic and heldout DSM/Stein checks are prepared;
no molecular actor update has been made. See notes/endpoint_entropy_candidate.md.

## Earlier frozen-path innovation protocol

A new bounded diagnostic keeps the16-step forward sampler and every endpoint
fixed while replacing the sequential backward Gaussian by a global Gaussian
in the path's original independent-noise coordinates. Exact linear-Gaussian
conditioning, nonlinear path-density transformations and directional gradients
pass6 targeted tests. This is a standard inference-based diagnostic, not a new
AI-method claim. Protocol: notes/innovation_posterior_diagnostic.md. The real
molecular smoke must pass before the32-path screen; no result is claimed yet.

## Latest completed screen: empirical weighted-FM projection failed its stop gate

All four runs and the256-attempt independent xTB assessment45812870 completed.
Source / uniform / linear / power post-refit ESS is1.119536 /1.000055 /1.000002 /
1.006283 of256. The prespecified all-student ESS<16/256 stop gate is triggered:
do not extend or scale this frozen-pool recipe. There is no new sampling gain.
This does not rule out every forward-learning or better-teacher approach.

All eight xTB panels converge32/32. ODE64 median strains are .62413/.68058/
.69270/.57412 eV; stochastic16 medians .62054/.58203/.67821/.60809 eV. These
successful geometry checks coexist with severe weight collapse. Stochastic
q90 strains worsen from source .789 to4.985/5.667/3.508 eV. On all256 samples,
stochastic overlap counts are0/2/3/7, so selected xTB convergence is not a
chemical-validity certificate. ODE32/64 discrepancies remain unresolved for
some paths (source worst atom1.886 A); no numerically converged ODE claim follows.

Evidence: evidence/forward_work_students_comparison_v1.json, generated by the
provenance-checked join. New campaign costs4096 teacher+128 smoke+2048 evaluation
oracle queries; inherited forward25024 and backward256 queries remain additional,
along with earlier development. All rejected submissions and source snapshots
are retained. The updated development appendix builds with9 main pages in
runs/verification/paper_20260910_students/main.pdf; formatting is not scientific
qualification.237 tests pass; research/NOVELTY_DECISION.md states the AI-method
priority and direct prior-art boundaries. Scientific submission readiness=false.

## Earlier completed frozen-forward diagnosis and forward-update protocol

Backward refit45802060 completed: 2048 fixed training paths, 500 backward-only
updates, 256 independent heldout oracle queries. The complete forward state and
heldout coordinates were unchanged. Before/after mean refitting, with original
and fitted scalar widths, heldout ESS remains1.0001--1.0016 of256. This rules out
this particular reverse-refit repair, not all possible reverse models. Source
and comparison hashes, all four arms and costs are preserved in
`evidence/backward_refit_5846_v1.json`. The smaller smoke45801586 is retained.

A prospective forward weighted-FM study is specified in
`notes/forward_mass_update_candidate.md`. The first step exports4096 independent
paths with complete densities and energies. Uniform, linear-mixture and
power-tempered empirical teacher weights will be compared. Stabilization is an
intermediate training objective, not target calibration. Weighting, tempering,
and flow matching already have direct prior art; novelty remains unestablished.

Teacher export45811812 completed with4096 new oracle queries: raw ESS2.625/4096.
Linear damping retains target fraction .025324; the matched-ESS power exponent
is .073005. Both stabilized ESS values are2048 by construction; neither makes
the teacher accurate at the full target. The raw concentration limits learning
from this pool and is retained in evidence/forward_work_teacher_5846_v1.json.
The full test suite passes237 tests. A bounded three-student/source comparison
with a predeclared stop gate is implemented. GPU smoke45812206 completed with
128 oracle queries, finite forward/reverse updates, independently rescored
forward densities within7.9e-5 nat and unchanged forward weights during reverse
refitting. Paired midpoint32/64 has RMS .0131 A and worst atom .2018 A; this is
a resolution warning, not a numerical convergence certificate. Production
separates training-selection and Gaussian RNG seeds explicitly. The smoke is
engineering validation only; its low ESS is retained.

Four-arm screen: source45812380 and uniform45812384 completed from1548e1f;
post-refit ESS1.11954 and1.000055 of256.
Linear/power v1 submissions were rejected by gpu_test QOSMaxSubmitJobPerUserLimit;
their submission-only directories remain intact. Linear45812690 and
power45812788 are running as v2 from9bb2f91, with unchanged training code and
recipe. Assessment45812870 is queued after all four completions. Every arm uses the same1000-step student recipe (source0) and
same500-step reverse refit. Independent256-attempt total xTB assessment and a
provenance-checked stop-decision join are prepared.

Earlier timestamped scheduler entries below are historical, not live state.

## September 10 UTC checkpoint

Three AgBr2 annealed-joint training seeds have completed: ESS 44.080, 51.204,
20.884 of 256. Against independent reference v2 their log-normalizer differences
are -0.05161, -0.15170, +0.15809 nat; empirical relative SEs are 13.73%, 12.52%,
21.01%. This supports further evaluation, not a replicated superiority claim.
Reference v2 uses eight scrambles of 8192 points, reuses 16,384 verified prefix
queries, and adds 49,176 oracle evaluations including checks. Its 300 K relative
SE is 1.82%; both reference versions and all seeds are retained.

Frozen-loader validation 45751790 completed and reproduced original paths
(max position difference 1.48e-7 A, work difference 9.35e-5). Fresh 4096-path
comparisons are submitted for seeds 9051/9052/9053: 45759407, 45759409, 45759411.
Fixed-joint and energy-only training 45750959/45750966 remains running, with
4096-path evaluations 45759434/45759436 depending on their successful completion.
Each new evaluation uses reference v2, batch 64, seed 9065 and separate cost.

Pure AgBr2 FM and short MALA controls completed. Short MALA still gives farthest
Ag-Br distance about 4.49 A versus reference about 2.54 A; it is not equilibrated.
Longer MALA 45759386 and HMC 45759394 have been submitted with 8,512 queries each
and chain-clustered uncertainty. Protocol: notes/long_chain_and_balance_protocol.md.
All seven new jobs use source ae63de3; do not resubmit live jobs.

New development baseline and xTB assessment completed (45747806/45749290), with
all 512 generated attempts plus eight references recorded. FM64 converges on
239/256 attempted relaxations; all eight references converge. No new-panel
physics-training result or reserved-condition method outcome exists yet.
The eight-atom ESS failure remains unresolved. Its joint-work fluctuations are
dominated by terminal reduced-energy spread, not only reverse-kernel error.

The first three fresh 4096-path evaluations have returned: ESS 581.18, 654.38,
353.98 for seeds 9051/9052/9053, with log-normalizer differences -0.08614, -0.05316,
-0.12530 nat (relative SE 3.84%, 3.58%, 5.08%, plus reference 1.82%). All estimates
are low relative to reference; this is not a calibration certificate. Evaluation
noise seed is shared across arms. Evidence: work300_large_eval_and_mcmc_v1.json.
Long HMC agrees more closely with reference distances (maximum Ag-Br 2.558 versus
2.537 A); xTB 32/32, median strain .64889 eV. Long MALA maximum distance is 3.826 A,
xTB 27/32, successful median .48722 eV. Retained MCMC draws remain correlated.
No superiority over HMC is established. A stale four-scramble prose limitation in
the saved comparisons is corrected in new code; the actual v2 reference has eight
scrambles and was used numerically. Historical outputs are preserved.

Second predeclared stream (9067) completed for all three annealed-joint seeds:
log-normalizer differences -0.02384, -0.01661, -0.00368 nat. Combining both streams
without selection gives ESS 802.65, 823.79, 640.28 of 8,192, log differences
-0.05450, -0.03472, -0.06264 and empirical relative SE 3.35%, 3.30%, 3.79%.
Shared reference error 1.82% remains additional. The first-stream offset is not
sufficient evidence of persistent bias. Full evidence: work300_two_fresh_streams_v1.json.

Real FlowMol observed-path batch values and gradients pass against independent
calls. Molecular log-variance smokes 45760184 (B2) and 45760450 (B16) completed;
B16 peak allocation 4.44 GiB, maximum score/sampler work difference 2.24e-4 nat.
500-update matched eight-atom run 45760898 is live; no performance result yet.
Its xTB assessment 45761719 is queued after successful training completion.
The unchanged HMC recipe is also submitted for eight atoms as 45761145, without
an unavailable normalizer reference. Latest paper build passes at 9 main pages;
this remains a scientific development/audit draft, not submission ready.

## September 10 UTC completed controls and next execution

All AgBr2 controls and their two 4096-path streams completed. Fixed joint:
combined ESS 886.45/8192, log-normalizer difference -.03239 nat. Annealed
energy-gradient control: ESS 13.72/8192, difference -5.06827 nat. The energy
ablation collapses here; fixed noise remains competitive with annealing.
Evidence: work300_all_triatomic_controls_v1.json, reproducible with
scripts/research/combine_work_evaluations.py. No HMC superiority claim follows.

Eight-atom LV 45760898 and xTB 45761719 completed and failed scientifically:
ESS 1/256, overlap 99.22%, xTB 25/32, successful median strain 316.30 eV versus
initial 32/32 and 4.405 eV. Frozen gradient diagnostic 45762271 completed with
256 oracle calls; scale-corrected forward covariance trace is about 37.6 times
the pathwise estimator's. Backward objectives differ, so variance is not a
uniquely established failure mechanism. NeurIPS 2025 arXiv:2506.10982 is direct
prior work. Evidence: work300_5846_lv_failure_and_gradient_v1.json.

Resume smoke 45763215 passed: max coordinate discrepancy 3.74e-7 A, work 1.80e-4
nat, original AdamW restored at step 500, temperature reset not repeated. Runs
45763360/45763361/45763362 continue fixed-joint/annealed-joint/energy to 1500
updates using source 4b4c615. Each adds 16,512 queries, cumulative 25,024.
The matched HMC run is 45763216. Three-arm xTB assessment 45764564 waits on successful completion.
Comparison audit 45775349 then verifies completed parent/continuation hashes,
update and query budgets, target equality, and all xTB denominators. No continuation performance result is available yet.

HMC-SMC reference pilot 45764408 uses source 73cee25, Gaussian starts, 32
particles, 128 fixed stages and eight leapfrog steps per stage. Budget 32,832
including terminal cache checks. It must pass the prespecified AgBr2 screen
against v2 quadrature before a larger-condition pilot; no converged reference
or new method is claimed. Protocol: notes/hmc_smc_reference_protocol.md.

HMC 25,024-query run 45763216 completed: xTB 32/32, median strain .56135 eV,
acceptance .9630. This remains a finite-chain control with eight independent
starts. HMC-SMC AgBr2 pilot 45764408 completed but failed its prespecified
normalizer screen: ESS 23.14/32, five ancestors (weighted ancestry ESS 1.326),
log-normalizer discrepancy +.71340 nat versus the .25 gate. Maximum Ag-Br
mean error .00945 A passes the geometric screen. Numerical cache check passes
at 8.80e-5 reduced-energy unit. Do not promote it as an eight-atom reference;
a single population does not identify persistent bias. Evidence: hmc_smc_reference_1137_screen_v1.json.

## Independent reference and compute checks, September 10 UTC

All four 32-particle HMC-SMC populations completed. Log-normalizer discrepancies
are +.71340, -1.84187, +.80471, +.89871 nat. Their pooled normalizer ratio to QMC
is 1.723 with descriptive SE .528 (relative 30.7%); no passing seed is selected
and this is not a certified reference. Evidence: hmc_smc_reference_four_seeds_v1.json.

Direct product Gauss-Legendre integration over labelled r01,r02 in [2,3.2] A and
cosine in [-1,1] provides a separate same-box check of the volume factors. Orders
20/32 differ from restricted QMC by +.04961/+.005303 nat; the Gaussian-volume
analytic test passes. This does not support a factor-of-two volume error.
Order 48, job 45770866, completed: difference to QMC +.007148 nat and
32-to-48 change +.001845 nat. All three orders use 151,360 new queries.
Evidence: triatomic_box_crosscheck_v1.json. The finite box does not certify
the full target.

Actual allocation/query ledger: completed_work_compute_ledger_v1.json. Same
8,512-query eight-atom mean-work runs use about 48 GPU allocation minutes,
versus HMC about four CPU allocation minutes, with different output statistics.
MIG/full-device labels are retained and not equated. No total-compute advantage
is claimed. The shared upstream model-training costs are not reconstructed.

A Gaussian location-mixture population-ESS ceiling is derived in
notes/gaussian_terminal_noise_ess.md and checked against direct density integrals.
This uses standard Gaussian/convexity identities; no novelty claim is made.
Positive-curvature local surrogates give fixed-noise ceilings .29--.38, but these
are NOT bounds on the non-Gaussian molecular target. The two new standalone
quadrature/noise-limit tests pass in addition to the last 222-test full suite.

## Completed 1500-step decision and new candidate

All eight-atom continuations, xTB 45764564 and audit 45775349 completed. Fixed /
annealed joint / annealed energy have ESS 1.174 / 2.218 / 1.092 of 256 despite
25,024 total queries each. All three xTB panels converge 32/32 with successful
median strain .49357 / .52856 / .13984 eV; matched HMC is .56135 eV. Annealed
joint work standard deviation falls to 19.44, but its restrained energy SD remains
.4703 eV and distributional calibration is unresolved. Further isotropic training
is not the current remedy. Full audit: work300_5846_1500_comparison.json.

User explicitly requested developing a stronger AI contribution. A candidate
learns bounded pairwise precision for correlated Gaussian path noise. Exact
sampling/log-density and covariance gradients are implemented; 228 tests pass.
This uses established elastic-network/anisotropic-noise ideas, not a claimed new
identity. AniDS and Chroma are prior art. The dense 1/N-normalized prototype has
unproven large-system scaling. Protocol: notes/pair_precision_candidate.md.

Real molecular gates passed: 45779357 (B2), 45780203 (B16), and 45780207 (frozen
loader). The precision head updates, B16 peak GPU allocation .443 GiB, and the
loader reproduces coordinates exactly and work within .000116 nat.
Live 500-step, 8,512-query controls from source ec121f1 are 45780761 (learned),
45780762 (fixed geometry), 45780775 (isotropic trace control). They share FM
initialization, seed 9051, target and mean-work objective. Check learned/fixed
initial equivalence and retain all outcomes; xTB 45781150 waits for all three.
Learned/fixed initial samples agree within 2.17e-6 A and .00181 work units;
see pair_precision_initial_equivalence_v1.json. No molecular benefit, originality
or ICLR readiness is established by these execution gates.

## Completed pair-precision candidate decision

All three 500-step arms and xTB 45781150 completed. Learned / fixed-rule /
isotropic-trace controls have ESS 1.130 / 1.230 / 1.000 of256. xTB convergence
is32/32,31/32,32/32; successful median strains .77571,1.00995,.88126 eV.
The learned arm improves on the fixed-rule variant but does not outperform
the earlier 8,512-query isotropic mean-work strain .56999 eV. Same-query HMC
is .90211 eV in one comparison with correlated samples; the .56135-eV HMC
result uses 25,024 queries and must not be called the same-budget baseline.
No calibrated or repeated sampling advantage is established. Preserve this
failed recipe rather than scaling it up. See pair_precision_full_comparison_v1.json.

On the first32 final geometries evaluated at t=15/16, learned relative-precision
maximum eigenvalue is1.03366, trace1.00198, fixed6.15194. The learned module is
nearly isotropic at these inputs. This is not a full-trajectory or causal test.
Next work should isolate forward coverage versus auxiliary reverse-kernel
limitations and examine moment/tail conditions before proposing another long
training run. Bounded covariance eigenvalues alone do not guarantee finite
path-weight variance. No specific causal explanation is yet established.

## What is established

- 228 tests pass, including real FlowMol parameter gradients, full-state and
  prior differentiation, exact discrete-adjoint comparisons, COM density,
  conditional FM targets, stochastic-replica identities, smooth geometry and
  dedicated Gaussian/Rademacher probe streams for common/independent controls,
  finite-path work identities, AIS, weighted CFM and intrinsic molecular proposals.
- A separate equivariant pair-kernel reference has a tested analytic divergence,
  including its coordinate and parameter gradients, collision derivatives,
  symmetries, checkpoint loading and discrete-adjoint gradients. It is prior-art
  architecture for a cheaper exact-trace comparison, with unproven molecular
  capacity. See `notes/radial_reference_protocol.md`.
- Legacy scalar ordering cannot be promoted to likelihood or Boltzmann sampling.
  The old routine integrates an endpoint prediction as velocity and freezes
  trajectory gradients. It is retained only for historical reproduction.
- The conditional factorization is `pi_frozen(m) q_theta,T(x|m)`. The positional
  checkpoint has no history, prior alignment or sampling retractions. Its
  changed shared backbone must not supply the frozen composition factor.
- Off-policy log-dispersion and independent-product identities are prior work
  or standard mathematics. They are not sufficient novelty. See
  `notes/literature_positioning.md` and `notes/jacobian_noise_bias.md`.

## Completed experiments

| Experiment | Result | Run / job |
|---|---|---|
| Original corrected-energy smoke | 20 updates, 20 finite energy contributions, 5.38 GiB peak | `runtime_smoke_v3`, 45570868 |
| Equal two-trace-budget controls | Squared/product × independent/common, 20 updates each, finite; independent arms skip one outlier each | `estimator_smokes_v1`, 45572136 |
| Affine Gaussian mechanism, five seeds | Mean exact KL: squared independent 0.74756; product 0.00138; common, exact and rotated controls reach zero | `toy_trace_v1`, 45572134 |
| Nonlinear shear mechanism, five seeds | Mean exact KL: squared Rad. 0.08547; product Rad. 0.00259; common squared 0.08141; common product 0.03745 with substantial variance | `toy_nonlinear_v1`, 45578222 |
| Original density resolution panel | Two of eight parents drift by 90.36 and 66.18 nats at 64→128 steps | `numerical_panel_v1`, 45569978 |
| Independent adaptive reference | Easy case succeeds at two tolerances; midpoint-128 differs by 0.2792 nats; both hard cases exceed 6,000 evaluations at both tolerances | `adaptive_panel_v3`, 45577793 |
| Conditional FM, endpoint head, T=0.8 | 1,000 updates, finite, 104.2 s, 5.06 GiB | `position_cfm_v1`, 45575822 |
| Conditional FM, endpoint head, T=0.8 | 10,000 updates, finite, 1,036.1 s, 5.11 GiB | `position_cfm_10000_s9004_v2`, 45580289 |
| Independent xTB, warm / 1,000 FM updates | 29/32 / 30/32 relaxations converge; successful-only median strain 13.09 / 5.01 eV | `position_xtb_v1`, 45577833 |
| Independent xTB, 10,000 FM updates | 31/32 converge; successful-only median strain 4.49 eV | `position_assessment_10000_v1`, 45582154 |
| 1,000-update density, full five sigmas | Two of eight parents still drift 82.74 and 112.61 nats at 64→128 | `position_density_v1`, 45577827 |
| 10,000-update density, sigmas 0.03/0.06 | Two of eight parents fail 0.1-nat gate at 128→256: 0.2561 and 0.4563 nats | `local_density_mature_v2`, 45582155 |
| Smoothed geometry counterfactual, rho=0.1 | Full-panel worst 64→128 drift falls to 2.5136 nats, still not converged; this changes the field without retraining | `smooth_density_v3`, 45584305 |
| Checkpointed high-resolution energy | Four updates, 128 steps, two local siblings, 609 s, 19.03 GiB | `highres_energy_smoke_v1`, 45581074 |
| Exact discrete-adjoint high-resolution energy | Same four updates, 275 s, 1.69 GiB; relative L2 final-weight difference 3.18e-8 from checkpointed run | `adjoint_energy_smoke_v1`, 45583400 |
| Displacement velocity, T=1, rho=0 | 10,000 FM-only updates; xTB 31/32 converge; median successful strain 4.14 eV; five of eight local density parents fail 0.1-nat gate, maximum drift 40.98 nats | `displacement_development_rho0_v1`, 45586851 |
| Displacement velocity, T=1, rho=0.1 | Matched 10,000 updates; xTB 30/32 converge; median successful strain 3.98 eV; five of eight density parents fail, maximum drift 0.4408 nats | `displacement_development_rho01_v2`, 45586884 |
| Fixed-weight RK4, rho=0 / 0.1 | At 32-to-64 steps, two / one of eight parents fail 0.1-nat screen; worst drift 18.2032 / 0.5573 nats | `rk4_density_rho0_v1`, 45595533; `rk4_density_rho01_v1`, 45595874 |
| RK4 common Gaussian energy, eight parents | Two updates, both contributions applied, 220.1 s, 3.78 GiB; weighted energy gradient norms 0.989 / 0.331 versus FM 3.090 / 4.417 | `rk4_energy_b8_pilot_v1`, 45599167 |
| Analytic-divergence radial reference, 10k / 100k FM updates | xTB 16/32 / 19/32 succeed; successful-only median strain 38.23 / 132.74 eV; all eight exact-trace 32/64 checks below 0.01 nat; poor generation retained | `radial_reference_development_v1`, 45601575; `radial_reference_development_100k_v1`, 45602524 |
| Radial exact/noisy controls | Seven energy arms each apply 20/20 contributions with no skips; exact value/shuffle/zero and squared/product independent/common; FM-only also finite | `radial_estimator_smokes_v1`, 45602694 |
| Finite-work/AIS toy, five seeds | Target basin mass .8: raw .5089, energy-only .8039, AIS .7982; within-mode spread target .5: raw 1.9616, energy-only .3011, AIS .4854; weighted FM student mass .7630 versus .4996 | `nonequilibrium_toy_v1`, 45608091 |
| Frozen molecular FM work proposal | 32/32 eSEN evaluations finite; ESS 1.655/32, maximum weight 72.87%; executable but inefficient on the declared restrained target | `molecular_work_pilot_v1`, 45608819 |

All molecular training above uses a single seed; warm versus 1,000 versus
10,000 updates is not seed replication. Warm q_0.8 was not trained for the new
conditional path, so its sampling comparison is an initialization diagnostic,
not a fair comparison with the original joint generator. The same eight
conditions and four priors per condition are reused. Reference structures
converge 8/8 with median strain 1.07 eV. xTB uses the recorded interior charge
and declared minimum electron-parity spin; those historical runs preceded recovery
of original DFT spin. The later electronic-state panel uses original spin.
Failure-conditioned medians alone do not establish quality, connectivity,
diversity or Boltzmann populations. The 10,000-step sampler's maximum
64→128 coordinate RMS drift is 0.0930 Angstrom.

The adjoint computes the exact first parameter gradient of the discretized
objective by saving states and recomputing each step, rather than inverting the
trajectory. It is an engineering improvement, not a new adjoint theorem. It
does not support arbitrary higher parameter derivatives. GPU agreement is
within floating-point error, not bitwise equality. See
`evidence/adjoint_comparison.json` and `notes/discrete_adjoint.md`.

The numerical panel uses probes fixed across time and solver resolutions;
training refreshes probes across steps. Noise estimates cannot be transferred
between these policies. Replica products correct stochastic squared-loss bias
on a fixed trajectory, not quadrature bias. Common probes are a required
control, not an omitted competitor. All Gaussian and unstable product toy
arms remain in `evidence/evidence.json` and the development appendix.

## New provenance and theory checks

Deterministic replay exactly reproduces all 150,000 training and 50,000 validation
perturbed geometries and atom/charge labels. In these two shards, perturbation
parent index equals the processed source row index. This recovers processed-tensor
provenance and the Gaussian perturbation mechanism, not raw OMol identifiers,
spin, unclipped charge or energy precision. See
`evidence/train_perturbation_provenance.json` and
`evidence/val_perturbation_provenance.json`.

The follow-up theory audit removes unsupported gradient-noise lower bounds,
corrects the endpoint-force conditional expectation, distinguishes discrete
quadrature from an exactly normalized discrete sampler, and replaces the old
capability matrix with the actual sampler/density scope. See
`audit/20260909/THEORY_FOLLOWUP.md`.

Classical RK4 is now available for both sampling and density, including full
discrete-adjoint gradients. It passes analytic fourth-order and real-network
gradient checks. Its extra trace evaluations are explicitly counted; it is not
an equal-cost replacement per step. The completed 16/32/64-step molecular panel
still fails its necessary screen. Moreover, midpoint-256 versus RK4-64 differs
by more than 0.1 nat on three / five parents (rho=0 / 0.1), although these two
resolutions both use 256 trace stages. See `evidence/solver_comparison.json`.
See `notes/rk4_density_protocol.md`.

The completed displacement runs each improve 15 paired xTB results and worsen
17 against the endpoint 10,000-update checkpoint, counting failures. Their
successful-only medians therefore do not establish better generation. Maximum
coordinate RMS drift is 0.0602 / 0.3576 Angstrom (rho=0 / 0.1). Comparing endpoint
and displacement runs changes both the regression target and terminal time.
See `evidence/displacement_comparison.json` for all parents and sources.

For difficult parent 1137, a separate CPU float32 coordinate-path diagnostic
finds minimum input atom distances 2.41e-5 / 0.00178 Angstrom (rho=0 / 0.1).
Near-collisions occur in the reverse trajectory itself as well as internal
network coordinates. This is a numerical diagnostic, not proof of a unique
failure cause; CPU and GPU trajectories can differ in roundoff.

## Latest completed numerical checks

- 45600126, `adaptive_displacement_rho01_v1`, source `26d75f5`: all six
  float64 DOP853 solves complete. For hard parent 1137, tighter tolerance still
  changes the mean centered density by 0.09306 nat and quadrature order by
  0.04008 nat. Successful ODE termination is not a density certificate.
- 45600490, `rk4_density_rho01_fine_v1`, source `227ffa3`: all eight
  RK4-128/256 comparisons complete. Parent 1137 drifts 0.45337 nat in the
  replica mean and 1.09298 nat in an individual replica; the other seven
  pass the necessary 0.1-nat mean screen. RK4-256 differs from the tighter
  adaptive references by 0.000244 / 0.07284 / 0.001172 nat for parents
  5846 / 1137 / 7544. Reference uncertainty still matters for 1137.

See `evidence/adaptive_reference_comparison.json`, which verifies complete
source panels, matching geometry/probe policies and source hashes.

## Non-equilibrium framework decision

Adopt the correct work interpretation and a separate finite-step correction
branch. The ideal residual `log q + beta U` is a generalized work; the legacy
off-policy group loss does not thereby become globally identifying. Noisy
unbiased log densities cannot be exponentiated into unbiased weights. These
repairs are documented in `notes/nonequilibrium_framework.md` and Appendix E.
SNF, AIS, FEAT and Microsoft's Enhanced Diffusion Sampling are explicit prior
work; their identities are not claimed as a novel contribution.

`cfm_mol/clamped_work.py` uses frozen conditional FM velocities and exact
Gaussian transition factors in an orthonormal COM-free basis. The proposal
stage runs in flowmol; the energy stage runs in omol25. The first molecular
pilot fixes charge/spin by an explicit convention and includes a harmonic
restraint in its target. Low ESS prevents a useful molecular claim. See
`evidence/molecular_work_pilot.json`. The five-seed toy and all controls are in
`evidence/nonequilibrium_toy.json`; FM distillation remains approximate.

The eight-parent energy run above is a runtime/gradient calibration, not an
energy-advantage experiment. It uses RK4-64, two common-within-parent Gaussian
replicas, the discrete adjoint and energy weight 0.001. No molecular performance
claim follows from its two finite updates.

## Data and evidence limits

A full composition census scans 3,902,107 archived training structures. There
are 6,527 validation structures with absent training stoichiometries, including
59 with at most 12 atoms and 304 with at most 24. Perturbation shards contain
14 and 70 matching parents at these limits. This is a conservative composition-
disjoint development pool, not a new blind test set. The old test file exactly
duplicates validation. Energy labels have already lost precision in float32;
charge clipping, missing source IDs and missing spin cannot be repaired from
those tensors. New loader code preserves float64 when supplied.

The eSEN checkpoint was found in woo_lab and verified by CPU H2 inference;
the first new molecular work pilot also succeeds in all 32 energy calls.
See `evidence/oracle_availability.json`. No checkpoint download or model token
is needed. The public 2025-05-14 training archive has now been recovered:
19,983,081,456 compressed bytes, SHA256
`1924f4f50128344cef731069b409757192a83eacdb47e0d0169efc3138ff9688`,
80 ASE-LMDB files and 3,986,754 raw records. Original electronic states and
float64 energies are present. Exact replay is restoring their links to the
3,941,522 accepted legacy records; an independent evaluation split remains
to be constructed and audited. See `evidence/raw_training_recovery.json`. The two environments remain separate. No home-directory writes or shared
FlowMol dependency edits are part of this takeover.

## Retained failed attempts

- 45568618: progress callback incompatible with disabled bar; fixed.
- 45569977: OOM from diagnostic force graphs and dynamic batching; fixed.
- 45570869: torch 2.2 mmap string-path requirement; fixed.
- 45574453: adaptive evaluation budget exceeded; initial script lost the first
  successful estimate. Subsequent code persists every solve or failure.
- 45575823: cancelled by this takeover after identifying boundary
  self-conditioning; partial results and stop reason retained.
- Several submissions rejected by the user's aggregate SLURM submit limit;
  these have null job IDs in `jobs.jsonl` and consumed no GPU runtime.
- `local_density_mature_v1`: launcher absent from source commit; fixed preflight.
- `displacement_development_rho01_v1`: submission saw an incompletely extracted
  source snapshot; no job created. Submission now publishes snapshots atomically.

## Required before submission claims

1. Numerical convergence of density and samples, then full-neighbourhood and
   stricter/adaptive confirmation; never remove failing parents after the fact.
2. Matched FM-only, squared, product, common, shuffled and zero-label controls,
   multiple training seeds and equal data/sample/compute budgets.
3. Independent potential evaluation and robust chemistry, diversity and mode
   coverage metrics. Strain alone is insufficient.
4. Preserved metadata and genuinely independent evaluation data, with a clearly
   specified conditional physical target and electronic state.
5. Demonstrated value against current energy-guided flow methods. The present
   code correctness and toy mechanism are foundations, not acceptance evidence.

Official deadlines: September 18 (abstract), September 25 (paper), 23:59 AoE.
https://iclr.cc/Conferences/2027/AuthorGuidelines . The existing PDF is a
reviewable audit/development draft, not a submission-ready positive-result paper.


## September 9 reconstruction follow-up

- 45648858 completed: four matched-oracle tempered-SMC arms on condition 5846.
  Endpoint ESS rises to 19.8--28.6/32, but ancestry remains 2--9 and the
  log-normalizer estimates disagree by up to 16.45 nats. This is not convergence.
- 45648859 completed: six work-path step/noise settings, with 32 target queries
  each. ESS remains 1.42--3.00/32. More steps or changed noise alone do not solve
  the finite-budget problem on this development condition.
- 45665442 / 45665466 completed: three seeds and four MALA proposal arms for
  conditions 5846 / 1137, including explicit rotational and permutation mixtures.
  On AgBr2, rotation and symmetry arms retain much better ancestry than the
  ordinary narrow mixture; on the eight-atom condition ancestry still collapses.
  All controls and seeds are retained in the corresponding evidence JSON files.
- 45674693 / 45674710 completed: three-seed defensive-mixture comparisons on
  1137 / 5846. The broad component repairs a mathematical tail-coverage defect,
  but the eight-atom results still have low ancestry and variable normalizers.
  Bounded ideal weights do not imply a useful finite-budget result.

The new rotational density uses the matrix-Fisher normalizer with observed
quadrature refinement checks, not an optimally aligned Gaussian substituted
for a density. The defensive component matches the known harmonic confinement
Gaussian. Both use existing mathematical identities with explicit citations;
see `notes/rotation_mixture_protocol.md` and `notes/defensive_symmetry_proposal.md`.
No new theorem or molecular training advantage is claimed.

Historical launch notes (both jobs completed; final results below):

- 45665423, `raw_metadata_replay_v1`: complete raw-to-legacy exact replay,
  restoring source paths, true charge/spin and float64 energies without changing
  historical tensors. Over 2.8 million accepted records matched at last check.
  This original source snapshot uses SQLite WAL. Cross-host live SQL reading
  caused `OperationalError: locking protocol`; do not query that live database.
  Use the atomic progress record and only its verified NumPy prefix. The writer
  remains healthy. The working script now uses DELETE journaling for future
  runs; after the original job closes, export and integrity-check a read-only
  index before multi-host use. Never treat partially filled arrays as complete.
- 45674682, `triatomic_reference_1137_v1`: independent, rotation-reduced
  randomized-Sobol integration from analytic Gaussian proposals, without learned
  templates. It was moved from serial_requeue to test with a one-hour limit
  after a priority delay. The reference reports resolution changes and
  independent-scramble uncertainty; it is not exact thermodynamic truth.

Raw replay confirms that legacy validation 1137 is raw row 118392, a neutral
AgBr2 doublet. The new declared electronic state matches the original source.
The older singlet-only perturbation convention did not. Broader global
charge/spin conditioning and a final source-group split remain required work.


## Latest validated state (September 9 afternoon)

Raw replay and export are complete. All 3,941,522 accepted geometries match
legacy positions, types, charge features, forces and float32 energies bitwise.
Recovered sidecars retain original float64 energies and true electronic states.
There are 72,423 out-of-range legacy charges, 678,514 non-singlets and 403,895
states above minimum-parity multiplicity. Electron parity is valid in all
records. The read-only SQLite export passed integrity and exact split counts;
use its immutable read mode, not the original live WAL database. The earlier
cross-host WAL issue did not stop or invalidate the producer's complete replay.

The independent three-atom reference 45674682 completed 12,304 eSEN queries.
At 3,072 points per scramble (four scrambles), log(mean Z-hat)=144093.1275689,
relative SE across scrambles=0.01675. The final resolution change is 0.0014593
nat, but the scramble uncertainty is larger; this is a statistical reference.
Rotation-energy variation was 6.49e-6 eV.

Direct IS with 1,024 potential calls per seed on AgBr2 gives mean weight ESS
233.85 for the confinement Gaussian, 44.52 for the ordinary defensive mixture,
and 90.35 for the defensive symmetry mixture. Mean absolute log-normalizer
errors against the reference are 0.05897, 0.15055 and 0.09840 nat respectively.
Symmetry helps the mixture here, but does not beat the simple Gaussian baseline.
The eight-atom condition remains severely weight-degenerate.

The paired 10,000-update electronic-state FM continuations completed:
45693190 global-state, 45693192 legacy features. Independent xTB assessment
45696871 succeeds on 32/32 generated structures in both arms. Median strain is
3.94384 eV (global) versus 3.89309 eV (legacy continuation), with means 5.81105
and 5.75158 eV. This confirms a correct, usable state interface, not a quality
advantage. The assessment uses original source multiplicities for both arms.
It remains a single-training-seed development comparison.

The method therefore remains scientifically not submission ready. The current
candidate needs repeatable value beyond the simple baselines; theory repairs,
metadata recovery and a larger test count are not that evidence.

Historical launch notes (jobs and archive recovery completed; results below):

- 45696874: refine all independent FM pilot centers by 50 monotone steps, then
  use the normalized defensive proposals; optimization is proposal construction,
  not equilibrium sampling. All centers remain. Initial pilot improvement averages
  5.32 log-target units; weight efficiency must still be evaluated.
- 45696876: equal-oracle direct-IS control with 1,568 samples per seed. Counting
  1,632 center-refinement queries once plus three times 1,024 endpoint queries
  gives 4,704 calls, equal to three times 1,568 in this control. Extra model and
  density costs are still separate and must be counted.
- Official validation archive download/extraction is active under
  `/n/holylabs/woo_lab/Lab/yulili/bgfm/raw_data/omol25/v250514/official_validation`.
  URL is the public Meta 250514/val.tar.gz object, verified HTTP 200 and
  21,293,980,744 bytes. Audit actual contents and train overlap before using it
  as an independent evaluation source; do not equate its name with a blind test.
- Finish assessment summaries/plots, freeze a final evaluation protocol, and
  select a scientifically useful method before investing in broad physics-student
  training. Global charge/spin baseline training is data FM, not Boltzmann training.
- `scripts/research/build_perturbation_shard.py` is an older unvalidated draft;
  none of the new reported experiments uses it. Validate or archive it before use.


All proposal-refinement and equal-oracle-budget jobs have now completed.
The refined eight-atom proposals still do not provide a stable high-ESS
population across seeds; retain these negative controls before further
method selection. Evidence is in `refined_importance_5846_v1.json` and
`direct_importance_budget_5846_v1.json`. The active long-term goal records
the user's explicit ICLR objective; it is not marked achieved.


The official validation archive has now also completed download, checksum and
extraction: 2,762,021 records in 80 ASE-LMDB files. See
`evidence/official_validation_recovery.json`. Its contents have not been used
for method outcome evaluation; source/composition overlap and an evaluation
manifest are the next required checks. `research/NEXT.md` records the next
research decision and the remaining negative evidence.


## Official evaluation and global-refresh protocol

The previous goal turn made concrete progress; no blocker was declared.
Current source 4c61d84 adds a fixed-q0 independence-MH refresh and a matched
hybrid (one global + one MALA) mutation. It is standard MCMC. Initial ancestry
is retained, and separate accepted-proposal IDs are not called independent
samples. The full suite passes 168 tests. A real eight-atom preflight completes
both hybrid arms with 20 potential queries each; it is an interface check.

- 45706102, `official_validation_audit_v1`, completed the full official
  2,762,021-record validation audit. It compares against 1,087,994 old training
  compositions, 28,453 old development compositions and 4,313,500 explicit
  source/reference-link hashes. Input-file hashes are checked. Candidate panels
  use a fixed hash of composition for development/reserved partition assignment
  and fixed within-stratum hash ranking, never energy or method outcomes. The
  completed 1,000-row preflight retains 967 candidates and excludes 33 by atom
  range, with no observed old-composition/source overlap. Full results are given
  below; do not use reserved candidate outcomes before protocol freeze.
- 45706103, `hybrid_smc_5846_v1`, completed three seeds of four matched arms:
  confinement or defensive-symmetry prior, each with two MALA moves or one
  independence-MH move followed by MALA. There are 64 particles, 16 fixed stages
  and 2,112 potential calls per arm/seed. Early global acceptance can still be
  zero at the final bridge; no improvement is yet established.

The unused perturbation-shard draft was preserved as
`notes/archive/build_perturbation_shard_unvalidated.py` and removed from active
research entry points. It remains explicitly unvalidated.

RegFlow (arXiv:2506.01158) is now added to required prior-work comparisons.
Regression-training an exact-likelihood invertible student is not a new idea.
A mean-work-trained stochastic teacher is another possible existing-method
baseline, now implemented with calibration results below. Prefer evidence-driven method
selection over accumulating architectural changes without molecular gains.


## Completed official-data audit and next physics-teacher calibration

Official validation audit 45706102 completed all 2,762,021 records. There are
2,564,135 eligible records with 2--200 atoms; 197,886 are outside that atom range.
No eligible record overlapped the old training/development elemental composition
sets or the checked explicit source links, and no missing/invalid electronic
state or nonfinite label was found. Fixed hash partitions yield 512,042 new
potential development records and 2,052,093 reserved records. Fixed stratified
reservoirs select 664 development and 722 reserved conditions; their elemental
composition sets are disjoint. The full candidate manifests and source hashes
are committed. Reserved conditions have not received method outcome queries.
Explicit calculation/reference links do not exhaust every parent-trajectory
relationship; keep this limitation rather than claiming more than the audit.

Global-refresh SMC 45706103 completed all 12 arms (four methods, three seeds).
It did not give a stable improvement on the eight-atom condition. Mean initial
ancestral ESS for confinement MALA/hybrid is 2.66/1.58, and for defensive-symmetry
MALA/hybrid is 2.18/1.38. Log-normalizer ranges remain 2.44--4.50 nats; final
independence acceptance is often zero. These are negative controls, not a
successful main method. Initial ancestry alone is not a mixing certificate.

A trainable finite-path mean-work module now provides full forward/backward
parameter gradients and external oracle-force derivatives, without neural
Jacobian traces. It is an existing path-KL/SNF objective, not a new identity.
The Euler/native-mismatch preflight gave poor geometries at both four and sixteen
steps. Plain and checkpointed two-update runs have identical parameter tensors.
An exact Gaussian reference bridge now keeps the native unit-width input while
matching the confinement endpoint width; zero neural residual gives constant
Gaussian work at any tested step count. The neural residual is smoothly bounded.

The joint and backward-only jobs 45716497 / 45716504 both completed 100 updates,
with 16 path steps and two paths per update. Joint training lowers held-noise mean
energy by 13.6534 eV and work standard deviation from 19.798 to 7.424, but weight
ESS remains only 3.485/64. Unweighted overlap rises from 12.5% to 17.1875%.
Backward-only training changes no forward sample coordinate (bitwise checked),
although work standard deviation falls to 13.189. Its ESS is 2.666/64.
These are a single-seed, single-condition teacher calibration, not a molecular
sampling success or final FM-student performance claim. Each used 328 oracle
queries including evaluation. See evidence/mean_work_calibration.json.

The next matched control removes path-factor gradients to the forward network,
leaving only terminal-energy gradients there while retaining backward conditional
likelihood training. Work values and forward simulation are unchanged at fixed
parameters. Tests compare forward gradients with energy-only autograd and
backward gradients with the full objective, with/without checkpointing.
This intentional ablation is not the full mean-work gradient.

The full suite passes 209 tests. Batched oracle inference agrees with serial ASE
on 32 generated geometries: maximum energy difference 8.87e-6 eV and force
component difference 5.04e-4 eV/A. Warm measured times were 75.88 s serial versus
6.99 s with batch 16 (10.86x in this single timing, not a general speed guarantee).
A separate four-geometry directional force check agrees within .00130 eV/A at
h=.01 A; smaller h shows worse finite-precision cancellation. Neither check is
independent physical-accuracy validation. Both reports are retained in evidence.

Condition manifests can now construct graphs without reference geometries.
Placeholder coordinates are rejected as FM data targets; reserved outcome use
requires a frozen method/manifest protocol. No reserved condition has received
method outcome queries. See notes/mean_work_training_protocol.md for the next
500-update, batch-16 matched experiment (8,512 potential queries per arm).
The two jobs are 45726541 (joint) and 45726651 (energy-gradient control), source
a76313a. Both were RUNNING at the September 9 20:13 UTC scheduler check; each
has a two-hour GPU limit. Their scientific results are still pending.

## Independent work-sampler assessment and condition-only inputs

Job 45728505 completed the frozen xTB protocol on the 100-update calibration.
For 32 outcome-independent sample indices per arm, initial samples converge
0/32 and jointly trained samples 6/32. Initial single-point SCC fails 29/32;
joint single-point SCC fails 24/32. The other failures occur in relaxation.
The six successful joint relaxations have median strain 12.7574 eV, a strongly
selected subset, not a whole-panel quality result. All failures and raw logs
are retained. Both complete 64-sample panels have multiple contact components;
median counts fall from six to five. Their element-pair-profile dispersion also
falls (median pair profile RMS 4.7846 to 2.6505 A). These distance diagnostics
are not chemical bonds or mode labels, and fragments are allowed by the target.
See evidence/mean_work_independent_assessment.json. This evidence strengthens
the need for matched structure checks, rather than establishing sampling success.

A pure FM control is now specified on this same condition and same intrinsic
Gaussian seeds. Earlier FM xTB panels used other conditions and cannot be
compared directly. Midpoint-16 and midpoint-64 use 32 and 128 field calls per
sample; likelihood and importance weights are unavailable and must stay null.
This is a diagnostic of stochastic-bridge initialization, not an equal-training-
cost Boltzmann baseline. The same fixed-index xTB protocol applies.

The condition-only graph constructor now respects FlowMol's upper-then-reversed
edge ordering. The earlier row-major constructor was not used in any method
outcome experiment; a regression test checks batching and reverse partners.
The work trainer now accepts audited development manifests without opening a
reference-coordinate dataset. A two-update CPU smoke on development row 167
(PbCl2, charge zero, singlet), four transitions and 12 oracle queries completes
with finite gradients and saved parameters. Its four-particle ESS remains about
one; it is an interface check, not a performance result. Reserved conditions
remain unqueried. See evidence/official_condition_work_smoke.json.

The actual source reference for condition 5846 passes the same xTB check: strain
.89014 eV and initial maximum force 2.8724 eV/A, with one contact component and
no overlap. This does not establish DFT accuracy, but shows the xTB failures are
not inevitable for this condition/electronic state. The reference is used only
as an assessment control, never as the initialization for generated positions.

The pure FM control's first submission was rejected by gpu_test's per-user
submission limit and created no job. Retry 45729734 uses the regular gpu
partition, the same half-hour limit, and immutable source 9f99c7c. It was PENDING
for priority at the September 9 20:32 UTC check. Do not relaunch this accepted job.
It started RUNNING by 20:34 UTC. The 500-update independent assessment 45729911
is queued with afterok dependencies on both training jobs, using source bf7669a
and the unchanged 32-index xTB protocol. All 256 endpoint geometries per arm
enter its geometry diagnostics. No result from these pending runs is yet claimed.

The same-condition FM control 45729734 subsequently completed in 3m31s.
Midpoint-16/64 xTB convergence is 29/32 and 30/32; successful-only median strain
is 4.76794/4.67530 eV. Both have one overlapping geometry among 64, versus 8/11
in the initial/100-update work panels. The maximum 16-to-64 coordinate change
is .66522 A, so numerical sample convergence remains unproven. No FM density
or importance weights were evaluated. See evidence/work_fm_control.json.

The comparison identifies damage from stochastic-bridge initialization, rather
than an inability of the original FM checkpoint to produce useful geometries
on this condition. A prospective native-mean option now compensates the reference
linear expansion before bounding the residual. Actual Gaussian transition
densities, target, noise and electronic state remain specified. Its quality
requires a new two-update preflight; finite gradients alone are insufficient.
See notes/mean_work_training_protocol.md. The existing 500-update reference-mean
controls are retained and continue to their predeclared assessments.
Native-mean preflight 45730876 (source a2f4a7b) was RUNNING at 20:43 UTC, with
two updates and 132 total potential queries prescribed. Its fixed-index xTB
assessment 45730934 is queued after successful completion. These outcomes are
pending. The full suite passes 209 tests after the parameterization change.

Native-mean preflight 45730876 and its xTB assessment 45730934 have now completed.
Before training, xTB convergence recovers to 30/32, but successful-only median
strain is 16.7263 eV, overlaps occur in 21/64 geometries, and weight ESS is only
1.928/64. After two updates: 28/32, 16.9588 eV, 20/64 and ESS 1.781/64. Restored
evaluator convergence is not adequate geometry or useful sampling. This motivates
the predeclared native-mean 500-update joint/energy-gradient comparison, with
8,512 potential queries per arm and the same initial checkpoint, random streams,
noise, restraint, temperature and electronic state as the reference-mean runs.
See evidence/native_mean_preflight.json and notes/mean_work_training_protocol.md.

Assessment now compares explicit physical conditions (ordered atomic numbers,
total charge and multiplicity), preserving each arm's complete source metadata.
This permits comparison across provenance-schema updates without dropping or
misidentifying electronic state. Missing FM weights remain null, never fabricated.

## Completed reference-mean 500-update comparison

Jobs 45726541 / 45726651 and assessment 45729911 completed. Both learning arms
used 8,512 potential queries and identical initialization/noise/settings except
the gradient ablation. Joint mean-work training ends with ESS 6.894/256, mean
energy -20907.189 eV, overlap incidence 7/256 and xTB convergence 12/32. The
energy-gradient arm ends with ESS 1.00036/256, mean energy -20919.395 eV, overlap
9/256 and xTB convergence 32/32. Its maximum importance weight is .999822.
Successful-only median strain is 15.8157 eV (joint) versus 6.76552 eV (energy).
In the failure-inclusive paired ranking, energy improves 31 cases and worsens
one against joint. No method achieves adequate geometry and sampling together.
The better work ESS is not sufficient novelty or practical sampling efficiency.

Native-mean 500-update jobs 45732136 / 45732178 are running from source c37c8d9;
their assessment 45732403 is queued after both. Each retains the same 8,512-query
budget. A direct MALA baseline from the 64 fixed FM16 samples is now implemented:
132 moves plus initial queries also totals 8,512 calls. It preserves rejected
proposals in query counts and does not call finite-time MCMC equilibrated.
The suite passes 209 tests, including clipped-drift MH Gaussian stationarity and
cached-value consistency. The new rendering script assembles completed results
with source hashes, failure denominators and explicit missing FM importance ESS.

## Direct MALA and target-temperature audit

FM-initialized MALA 45733994 completed 132 moves for 64 chains: exactly 8,512
potential queries, 228.4 s for sampling and 4m41s for the complete job. Acceptance
is .7139, but this is not a mixing certificate. Mean eSEN energy rises from
-20920.795 to -20916.673 eV. xTB convergence falls from 29/32 to 25/32 and
successful-only median strain rises from 4.76794 to 9.58024 eV. In paired ranking,
three cases improve, 27 worsen, two fail both. No equilibrium density or ESS is
claimed. This is a direct, matched-oracle baseline, with additional FM setup
cost and no learned-model amortization. See evidence/fm_initialized_mala.json.

These results also expose a target-definition issue. The existing kT=1-eV
calibration corresponds to about 11,604.5 K. Independent AgBr2 quadrature for
that target gives mean farthest Ag-Br distance 7.0083 A and radius of gyration
3.4852 A, versus 2.4972 and 1.9369 A for the recorded OMol structure. These are
means, not contact/dissociation probabilities or proof of eight-atom mixing.
Compactness and xTB relaxation are not direct tests of sampling this hot eSEN
target correctly. Earlier structure comparisons diagnose generation relevance;
they must not alone be used to declare a Boltzmann sampler mathematically wrong.

A prospective, fixed diagnostic now compares 300-K and 1000-K MALA targets, using
identical FM initial geometries, state, oracle, restraint, seed and 8,512-query
budget. Proposal variance scales with kT; score cap scales inversely so capped
physical-force drift stays fixed. This tests target suitability/local relaxation,
not final mode populations. No untrained FM temperature feature is changed.
Keep every 1-eV result. See notes/target_temperature_audit.md and its evidence.

The 300-K/1000-K diagnostic jobs are 45735846/45735854 (source f4a8a08), both
RUNNING at September 9 21:23 UTC. Their launchers include independent xTB
assessment. Native-mean training and its dependent assessment remain active.
The full suite passes 209 tests, including cold-target Gaussian MALA invariance.

## Physical-temperature results and terminal-noise resolution

The 300-K/1000-K MALA diagnostics completed in 4m00s/4m07s. Both used 8,512
queries and the same initial FM geometries. At 300 K, xTB convergence is 31/32,
median successful strain 2.40954 eV, with 30 paired improvements, 1 worsening and
1 both-failed. At 1000 K: 30/32, 2.07368 eV, 29 improvements, 2 worsenings and 1
both-failed. These are improvements from a standard MALA baseline, not a new
learned-method result or proof of equilibrium populations.

The completed 1-eV native-mean 500-update comparison also remains limited.
Joint work: ESS 8.671/256, xTB 16/32, median successful strain 15.3193 eV.
Energy-gradient: ESS 1.352/256, xTB 32/32, median 4.97521 eV. Energy wins all 32
paired xTB rankings, but has concentrated importance weights. Both are retained
in evidence/native_mean_500.json; 1-eV generation quality and target accuracy
must remain distinct questions.

Independent cold-reference coverage has improved. Four new shape-QMC scrambles
with 4,096 points each give normalizer relative SE 4.18% at 300 K and 1.45% at 1000 K,
with minimum scramble ESS 116.6/632.2. This is statistical reference evidence,
not a global convergence certificate. The proposal Jacobian passed a full
six-variable check and known Gaussian integrals. New queries 16,408; the existing
12,304-query independent pilot is accounted separately. No learned FM templates
enter this reference. See evidence/refined_triatomic_reference.json.

A second-order Tweedie identity exposes a fixed terminal-noise restriction.
The current constant schedule permits exact target log-curvature only up to
40.050 eV/A2. Finite-force differences at four fixed cold-MALA endpoint indices
instead give maximum curvature 185--233 eV/A2, stable between h=.01/.003 A.
This supports a violated necessary local condition; it is not a global Hessian
certificate. A square-root noise schedule raises the ceiling to 640.050 eV/A2.
Actual Gaussian factors remain explicit and positive at every finite step.
See notes/terminal_noise_resolution.md; the identity and scheduling idea are
established mathematics, not claimed as new. Temperature-input initialization
now preserves the pretrained constant-input field before learning the new target.
The full suite passes 209 tests. A real 300-K CPU update check is finite but has
only 4 evaluation particles and ESS 1; it is not performance evidence.

Current 300-K training jobs 45744181/45744274/45744278/45744280 (source 71a853f)
were all RUNNING at 22:34 UTC. Explicitly labelled eight-atom assessment 45745430
and AgBr2 normalizer/moment/xTB assessment 45745431 are queued with afterok
requirements. No outcome from these ongoing runs is claimed. The comparison
code tests the energy-offset sign and rejects potential/restraint mismatches.
The full suite passes 209 tests. See NEXT for the exact active experiment map.

## First independent 300-K calibration and broader development panel

The initial 300-K jobs 45744181/45744274/45744278/45744280 and assessments 45745430/
45745431 are COMPLETE. AgBr2 annealed joint work, seed 9051, gives ESS 44.080/256,
maximum weight .06089 and log-normalizer difference -.00924 nat from the independent
reference. Empirical relative normalizer SE is 13.73%, versus 4.18% for the reference;
the close point estimate is not .9%-precision evidence. Weighted invariant moments
are consistent within these finite-sample error estimates. xTB improves from 28/32
to 32/32, with successful-only median strain 1.61084 to .45196 eV. This is one
three-atom condition and one training seed, not a general ICLR contribution.

The eight-atom case remains uncalibrated: fixed joint / annealed joint / annealed
energy have ESS 1.0003 / 1.0687 / 1.0006 out of 256. All three pass xTB 32/32, with
median strain .56999 / .94594 / .13002 eV. Better geometry and energy do not imply
correct target populations. These negative sampling results remain retained.

The fixed replication and ablation runs from source be9fb7a are 45750942 (annealed
joint, seed 9052), 45750951 (seed 9053), 45750959 (fixed joint, seed 9051), 45750966
(annealed energy-gradient, seed 9051). The replicas were RUNNING and the two
regular-GPU controls were PENDING for priority at 23:36 UTC. Frozen-loader check
45751790 is also pending; it must reproduce a saved evaluation-stream prefix
before the independent 4096-sample evaluation is used. Trained temperature
weights must not be reset a second time during inference.

An outcome-independent eight-condition development panel is frozen by hash and
replays byte-for-byte. It spans 9--20 atoms, neutral/charged and singlet/open-shell
strata, including Re and Pt. All eight raw reference identities/labels are checked;
all eight condition-only CPU interface tests complete. No reference coordinates
enter generation. Full baseline 45747806 and independent assessment 45749290 are
queued. The same-condition AgBr2 FM control 45747807 and MALA 45749291 are also
queued. Every failure is retained. Reserved evaluation data remains untouched.
The full suite passes 210 tests; the new frozen-inference loader still requires
its pending real-GPU numerical check.
