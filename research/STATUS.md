# Current research status — September 12, 2026

Authoritative checkpoint: `research/ACCEPTED_UTILITY_STATE_20260912.json`.
The ICLR goal remains active and scientifically unachieved. The latest bounded
geometric learner has a small offline proxy gain but no established
fresh-proposal sampling advantage. Do not scale its frozen weights.

## Implemented and audited

The bounded guide directly trains signed actual-MH accepted potential decrease
with query costs, using differentiable observed forward/reverse joint densities.
Production sampling remains no-grad. Its residual log-score bound B=0.5 gives
a complete coordinate-density ratio bound exp(2) relative to physical site arcs
on common support. Exact finite-state identities, actual-map finite differences
and independent density calculations qualify the implementation; they do not
establish method novelty or sampling efficiency.

The protected physical behavior dataset contains 1,671 attempts: 1,597 scored
and 74 retained failures. Its frozen split has 36 FIT and 12 internal-selection
parents across four fixed compositions. Data construction costs 12,288 trajectory
plus 5,822 preparation = 18,110 raw calls. Both 300-step training seeds use no
additional oracle calls. All final metrics replay, including trained-objective
finite differences. Two-seed training and its audit are complete.

## Offline proxy and fresh proposals

On the 12 internal-selection parents, estimated utility per expected raw call
rises from 0.003225565 to 0.003313737 / 0.003313180 eV, gains of 2.73% / 2.72%.
Descriptive parent-bootstrap intervals for differences are
[3.307e-5, 1.511e-4] / [2.758e-5, 1.554e-4] eV per call.
Scored-edge effective counts remain about 402 / 401 versus 406 at initialization;
maximum observed importance ratios are below 1.68. These are importance
diagnostics, not molecular ESS. Summary:
`runs/accepted_utility_summary_v1/results.json`.

The frozen fresh-proposal follow-up uses 72 source states (first/middle/last
joint-attempt sources from two physical trajectories per selection parent),
16 draws per state and three methods. All 3,456 attempts are retained.
Each method scores 1,118 candidates and costs 2,236 raw calls: 6,708 total.
All proposals and 3,354 MH ratios pass replay and independent checks.

Actual fresh-proposal utility per raw call is 0.002632881 for physics,
0.002635169 for learned seed 0 and 0.002632618 for seed 1. Differences are
+2.288e-6 and -2.625e-7 eV per call, with descriptive parent intervals
[-8.543e-5, 1.131e-4] and [-1.620e-4, 1.694e-4]. Neither establishes a gain.
These intervals do not rule out a small effect, and the selected source
population differs from the full offline population. Expected constitutional
flow is also similar. No complete-chain or equilibrium conclusion follows.
Summary: `runs/utility_onpolicy_summary_v1/results.json`.

## Preserved negatives and next work

The scalar work / work-plus-force learner's 19--24% prediction improvement did
not give molecular benefit. Its 48-parent, six-composition comparison loses after
model-data costs (+0.5453 / +0.4929 eV against physics); equal-inference intervals
span zero. Corrected trajectories use 66,588 raw calls; actual research cost is
67,980 after the retained numerical repair. All original failures and discarded
suffixes remain. See `research/CONDITIONAL_CHAIN_RESULT_20260912.json`.

The old vector learner's failures against concentration 64/400 controls and
six-composition transfer remain unchanged. Source support counts remain
45,32,106,99,0,19,0,96 out of 256 each. Condition 6 has empty connected support
under the pinned builder; condition 4 remains unresolved. This is algorithmic
support, not a physical chemistry impossibility claim.

Next: bounded conditional action selection together with placement, keeping the
move-family schedule fixed. The old selector's failure must inform this distinct
comparison. See `notes/bounded_action_geometry_candidate_v1.md` and NEXT.
No fitting on fresh validation outcomes, either evaluated molecular cohort or
the 722 reserved conditions.

All current BGFM jobs are terminal, verified with Slurm accounting:
46186496, 46186856, 46186968, 46189583 and 46189707. The manuscript includes the
offline signal and fresh-proposal null result:
`runs/verification/accepted_utility_20260912/main.pdf` (8 main pages, 20 total).
Build evidence: `research/evidence/accepted_utility_build_20260912.json`.
The paper is not scientifically submission ready.

Previous status: `notes/archive/status_through_scalar_chain_20260912.md`.
