# Next research decision — 2026-09-11

The ICLR goal remains ACTIVE and unachieved. The latest turn made substantial
progress: the original48-arm campaign closed, a new conditional proposal and
support-correct training gradient were implemented, physical reference failures
were diagnosed, and a reversible chemical exchange crossed the observed isomer
barrier in both repetitions. This is not yet a validated learned molecular method.

Read RECONSTRUCTION_CURRENT.md first. It replaces earlier optimistic static-
refinement assumptions and gives the mathematics, evidence limits and next work.
The primary manuscript still documents the earlier candidate; do not relabel its
unrestricted-target results as results for a hard validity-conditioned target.

## Latest action-policy checkpoint (September11)

The formerly requested learned chemical selector is implemented and trained.
Read `notes/chemical_policy.md` and `CHEMICAL_POLICY_STATE_20260911.json`.
The table job46038056 and two-seed training array46038463 are complete. Table
preparation uses5564 raw queries for training and404 for development warm-up;
all proposals, saved raw paired outputs and random streams replay. The model
has7106 parameters and improves its TRAINING utility from0.5222 to1.2441/1.2438.
This is not held-out sampling evidence.

Evaluation46039029 is submitted/running under the immutable v2 evaluation
protocol. It compares both learned policies to fixed0.5-local and fixed0.1-local
uniform action selectors. Learning saturated the0.1 local probability floor,
so the second baseline tests whether action selection adds anything beyond move
frequency. Uniform controls receive the full source/training preparation budget
as extra sampling; actual total-cost prefixes, not nominal endpoints, define
comparisons. Initial-source neural generation and offline wall times are separate.

Finish these six arms and run the recorded full replay audit before selecting a
route. Preserve every invalid proposal and the four-parent source denominator.
Do not expand this one-composition policy if it cannot beat the strong uniform
control after costs. Do not call increased training utility AI novelty, endpoint
energy reduction equilibrium sampling, or these four parents generalization.
The older immediate-next lists below are historical context where superseded.

## Completed original campaign

-48/48 qualified arms,0 failed,866304 raw refinement queries; source generation
 adds36864 and prior failures/other experiments remain additional.
-8/8 independent geometry conditions:1792 attempts,1666 converged,126 failed.
-Canonical evidence: evidence/parity_production_complete_v1.json and
 evidence/parity_geometry_complete_v1.json. Every model replays from its saved
 checkpoint. The earlier prefixes are preserved.
-The full/compact EACF and direct MALA/HMC comparisons remain adverse to the
 current convex-refiner superiority claim. Do not expand that capacity recipe.
-Source generation is still FM64 midpoint at T=1 with the displacement head,
 .025-A COM noise and fixed atom identities/electronic state. No source density
 or importance weights have been manufactured for this sampler.

## New route and experiments

1. ConditionalMolecularProposal acts on Gaussian noise conditioned on the FULL
 current geometry. It has exact intrinsic volume, differentiable forward/reverse
 proposal densities, low-rank collective updates and a same-network affine
 control. Source: cfm_mol/conditional_molecular_proposal.py. Geometric support and
 supported-MALA primitives are separately tested. Source entropy is not needed
 by the Metropolized kernel; the FM initializer can be improved separately.
2. Shape-jump training failed the mode-mixing screen. The oracle-coordinate
 diagnostic improved mode ESS but used privileged information. The spectral
 variant derives a slow direction from fixed radial features of training data,
 without mixture labels; it is not a general molecular representation or global
 spectral-gap certificate. Jobs45982929,45985015,45986980 and45989157 are complete.
 All are controlled Gaussian-mixture toys, not molecular success.
3. Frozen long sampling: nonlinear/local-MALA mixture has mode ESS264.5/842.7,
 MALA250.7/107.3, affine mixture274.5/209.6. Total comparison budget156704
 analytic target evaluations includes inherited training. Seed variation and
 Rhat>1.01 prevent a stable superiority claim. Source runs/frozen_proposal_long_v1.
 Keep the poor global-only result and every affine/RWM/MALA control.
4. Hard validity changes the target support. The old full-support diffeomorphic
 refiner would have infinite KL to a target that is zero outside that domain.
 Naive pathwise differentiation through an indicator misses boundary terms.
 metropolis_utility.py implements the fixed-proposal score gradient; exact
 boundary tests pass. It has not trained a molecular actor yet.

## Reference and chemistry evidence

-Unrestricted reference pilot45980521:65600 raw queries and complete proposal
 replay. Rhat2.03 for cold energy,2.62 for restrained energy; FM ladders remain
 fragmented, QC ladders connected. Not an equilibrium reference.
-Connected pilot45992214:41998 raw queries,11801 unsupported proposed states
 skipped before the oracle, complete proposal replay. Cold restrained-energy
 Rhat1.78; all retained states are connected but initialization memory remains.
 The failed predecessor45989633 stopped at a full node /tmp before Python,
 zero physical queries. Its logs/output remain. Runtime temp/cache directories
 now live under each run. Do not clean other users' node storage.
-Graph audit: condition0 has66/4096 admissible training and8/512 development
 geometries;34/66 and4/8 pass the current RDKit assignment. Those four development
 molecules have different constitutional connectivity from the QC reference.
 See geometric_domain_source_audit_v1.json and condition_00_bond_perception_audit_v2.json.
-The reference files are for assessment. QC-origin coordinates must not become
 generator/policy training inputs. FM-origin reference ladders never exchanged
 walkers with QC-origin ladders.

## Reversible chemical transition diagnostic

Job45999224 is COMPLETE. Same warm starts, two transition seeds,1544 new raw
queries per arm. Local/exchange/local cycles convert both FM-origin starts to
the lower-energy reference connectivity in both repetitions; three local-MALA
moves per cycle do not. The accepted maps carry covalent-radius scaling,
explicit inverse actions and the intrinsic Jacobian(s_i*s_j)^3. This uniform
physical move is a BASELINE, not a learned policy or established AI novelty.

Evidence: terminal_exchange_diagnostic_audit_v1.json. All1040 saved cycle
structures have checked identities, matched starts and COM. v1 does not save
all per-query forces, so do not claim complete stochastic replay. Strengthen
that trace before molecular policy-production comparisons.

## Immediate substantive next actions

1. Learn a permutation/rotation-invariant chemical action policy from generated
 TRAINING parents and new training transitions only. Compare against the uniform
 reversible chemical move with the SAME map, chemical checks and total cost.
 Record training, initialization, every invalid proposal and actual oracle calls.
2. Combine useful chemical edits with validated geometry moves. A local kernel
 remains essential; neither rare topology changes nor large shape jumps alone
 establish adequate sampling. More general fragment moves need their exact
 inverse, chemical eligibility, atom inventory and intrinsic Jacobian first.
3. Qualify independent distributions and electronic/chemical validity on further
 molecules and larger systems. Current reference pilots are not qualified.
 Use source-validity eligibility before energy outcomes; do not cherry-pick.
4. Rewrite the main paper around the final tested method once repeated molecular
 advantages exist. Correctness repairs, generic MH, mode jumps, accepted-jump
 objectives and standard density identities are not sufficient novelty claims.

Reserved722 outcomes remain untouched. Never write home, alter shared FlowMol,
merge the FlowMol/fairchem/JAX environments, or count formatting/tests as ICLR
readiness. The old score/CNF/work/auxiliary failures and the index-split symmetry
failure remain preserved. No unfinished research goal should be marked complete.
