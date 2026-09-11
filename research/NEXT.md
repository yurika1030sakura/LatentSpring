# Next research decision

**Post-handoff Claude update reviewed:** read `../notes/claude_update_review_v1.md` and
`research/evidence/claude_update_review_v1.json` (repository-relative). Fixed-index
splitting is a labelled diagnostic, not a permutation-equivariant repair.
Three loader-failed experiments were recovered without retraining; the two
colliding LJ13 arrays were stopped with evidence preserved. Condition0 of the
original molecular campaign is complete; refresh all other job states.

Claude continuation: start with `../CLAUDE_HANDOFF.md` and
`HANDOFF_STATE_20260910.json`. The handoff contains executable audit/assessment
commands, recovery rules and claim boundaries. It does not complete the ICLR goal.

Full ICLR goal remains ACTIVE and unachieved. Previous goal turn: PROGRESS.
We implemented exact-volume refinement, diagnosed/repaired a material source/
target symmetry issue, completed physical and end-to-end qualification, rewrote
the primary method manuscript, and launched corrected broader production.
294 tests pass. No broad calibrated sampler or qualified ICLR advantage yet.

## Current live campaign — re-query before acting

Latest follow-up: condition0 training45914819 is COMPLETED, six qualified arms.
Condition1 has also finished its allocation; newly completed arms still need
full replay. Array2 and7 were running at the last query;3--6 pending.
Geometry45938396 completes224 attempts (192 converged,32 failed). Read
evidence/parity_geometry_condition_00_v1.json. Optional index splitting remains
a labelled diagnostic. See the newer review above before the historical release
notes below. Upstream EACF model/spherical-layer smoke45939182 is RUNNING on CPU,
30-minute cap, output runs/eacf_upstream_smoke_v1. Its selected upstream tests
are engineering qualification only; inspect terminal state and junit.xml.

-45914819 parity_entropy_condition_00_v1: corrected condition0, all six arms,
 regular GPU,4-hour cap. Query its actual state; do not duplicate.
-45914826 parity_entropy_conditions_1_7_v1: array1--7, concurrency2,4 hours/task,
 same corrected protocol. Task7 was extended to6 hours based on measured20-atom
 CPU-oracle cost; other tasks remain4 hours. Array concurrency remains2.
 No dependency is needed: all source pools are complete.
-Full source45889306 completed8/8 conditions,36864 queries,2h06m37s.
 Source directory: runs/species_breadth_source_full_v2, one condition_XX folder
 with4096 training and512 development rows each. Raw energies are components,
 not already-projected target energies. Every condition passed exact source replay.
-The old raw-target jobs45892106/45892107 were cancelled BEFORE execution,
 after a temporary hold. Never unhold/relabel them as corrected training.

## Corrected source, target and estimator

Read notes/parity_refinement_candidate.md and
research/evidence/parity_training_protocol_v1.json FIRST.
The source FlowMol actually has n_cp_feats=4 and SO(3) cross-product features.
DO NOT infer that its distribution is O(3)-invariant from our adapter's symmetry.
An equivariant invertible adapter cannot remove a source symmetry defect.
Use q0_plus=(q0+inversion#q0)/2 and E_plus=(E(x)+E(-x))/2;
F_plus=(F(x)-F(-x))/2. The current target has no fixed-stereoisomer restriction.

Source inversion is a stochastic mixture, not a zero-entropy-change bijection.
The exact transport-KL baseline is q0_plus. The separate source-mixture KL gain
toward an even target is JS(q0,inversion#q0) in[0,log2], not estimated here.
The even transport bracket can be evaluated on original unflipped parents.
Old explicit works can be re-scored with a uniform-sign auxiliary whose factors
cancel: W_plus=W_raw+(E_plus-E_raw)/kT. No original source/reverse parity is needed.

Production uses1000 updates,16 independently inverted parents/update, one RAW
potential query/parent. Its expected energy/gradient equals the projected-target
objective because the refined law is invariant. Evaluation queries BOTH signs.
Each arm18048 raw queries:16000 training+1024 base+1024 adapted evaluation.
48 arms planned =866304 queries, plus source/pretraining/other costs. Actual
costs and failures must remain separate from these budgets. This one-query
linearity argument DOES NOT apply inside exponentials, FAB/AIS weights, MH
acceptance or projected-energy histograms. Those need actual E_plus or their
own separately justified estimator. No learned odd-energy control variate exists.

## Completed qualification and evidence

-Raw audit45899596:256 queries on4 geometries per8 conditions; mirror energy
 defects .0533--.2262 eV, rotation/permutation errors <5.73e-6/<3.58e-6 eV.
 Raw h=.001 force checks pass; h=.0003 failures on3,6,7 are retained.
-Projected audit45906924: all8 conditions pass640-query symmetry/force checks
 on FD ladder .003/.0015/.001 A. This is numerical qualification, not a global
 physical-accuracy or sampling certificate.
-N8 projected re-score45906933:14336 inverted-energy queries; same2048 parents.
 Convex-minus-affine -.013366+/-.002517 and-.012610+/-.002150 nat; convex-minus-
 typed -.048981+/-.017293 and-.038923+/-.016395. Signal survives, but every
 path ESS remainsabout1/2048. The parity repair did NOT solve that weight collapse.
 Source augmentation and target projection are explicit NEW protocols; old raw
 results are retained separately. Evidence: parity_refinement_rescore_v1.json.
-Corrected training smoke45911477 passes24/24 real cases,2304 raw queries,
 including loader replay on saved inverted samples. Earlier raw smoke45888794
 (24/24,1536 queries) remains distinct. Full repository suite294 tests passes.
-Affine xTB45899593 completes64/64; medians .593530/.591223 eV, close to convex
 controls. No measured geometry-superiority claim. See affine_context_xtb_v1.
-Failed bulk producer45885669 was cancelled14m12s after a4096-row CPU RPC
 timeout; attempted physical-query count0--8192, not zero cost. Replacement
 uses max32 structures/RPC and persistent coordinates/scored chunks. Original
 failures and raw/acknowledged counters remain retained.

## Next useful work

1. Monitor corrected production above, retaining all6 arms/condition and all
 failures. Paths: runs/parity_entropy_condition_00_v1/condition_00_METHOD_sR;
 runs/parity_entropy_conditions_1_7_v1/condition_XX/condition_XX_METHOD_sR.
 METHOD convex/affine/typed; R0/1. Use completed immutable prefixes while other
 arms run. Verify source/target/refinement hashes and actual query counts.
2. Broad comparison is now implemented in compare_parity_campaign.py and passes
 all24 real engineering-arm replays (parity_smoke_replay_audit_v2.json). Run it
 on completed immutable production arms, retaining unresolved/failed outcomes.
 Independent geometry/diversity/xTB orchestration and aggregation remain to do.
 Pair the same512 parents, separate training streams, and report per-internal-DOF
 as well as absolute changes. Do not invent importance weights for this new
 source or pool replicated rows as independent samples. assess_work_panel.py
 now supports strict shared-parent contracts and explicit manifest identities;
 its assessment/geometry focused tests pass3/3. Keep all failure denominators.
3. Strong external EACF/FAB and matched-HMC comparisons remain required. Upstream
 EACF is cloned unmodified at
 /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_20260910,
 commit beafab1b1ccd2b770572daeef1cf15f3fe199c21 (MIT). Dependencies are installed
 in external/eacf_env_20260910; core imports and pip check pass. No model,
 energy bridge, GPU runtime or baseline is qualified. The resolved jaxlib is
 0.4.13+cuda12.cudnn89 despite the historical CPU install-log name. FAB-JAX and
 Torch remain absent from this separate environment. Never alter FlowMol/fairchem.
 Read evidence/eacf_runtime_install_v1.json and the comparison contract;
 joint-KL bounds are not exact marginal errors. Installer has finished.
4. An odd learned energy control variate is only a prospective variance-reduction
 idea. Exact oddness guarantees zero mean under the corrected invariant law,
 independently of critic accuracy, but poor fits can increase variance. Pursue
 only after measuring odd-gradient variance and fixing matched-query validation.
 It is not the failed proposal-score actor and is not implemented.
5. Main manuscript now lives in paper/main.tex plus M1_refinement.tex and
 M2_refinement_proofs.tex, including parity repair and completed raw/projected
 controls. Latest build: runs/verification/paper_20260910_parity_final/main.pdf (6 main pages). Historical
 audit is paper/legacy_audit.tex; build.sh accepts it as a second argument.
 Continue updating the NEW main text rather than only the old appendix.

Reserved722 outcomes remain untouched. Broad performance, credible distribution
references, total-compute superiority and final reproducibility are unproven.
Do not declare ICLR readiness, complete the goal, or mark blocked from these
partial results. Never write home, change shared FlowMol, merge environments,
or omit the prior score/CNF/work/covariance/empirical-CFM/auxiliary failures.
