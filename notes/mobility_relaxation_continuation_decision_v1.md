# Mobility continuation and actual boundary-step decision

The previous goal turn was progress: it completed the original paired mobility
audit, changed the next diagnostic and updated the reproducible manuscript.
The present turn completes the frozen128-cap continuation and actual boundary
energy checks. The full ICLR objective remains active and unachieved.

## Completed continuation

All53 arms stopped by the32-query cap were continued, preserving original
optimizer histories, partial searches and all cached physical evaluations.
The other75 arms remained unchanged. The new cap is128 candidate evaluations
per arm. It adds3,768 raw calls against a frozen10,176 limit, for9,332 cumulative.
Producer46213860 and audit46213924 are COMPLETED 0:0. Every original query is
replayed without physical requery. All4,538 cumulative optimization trials pass
full replay and independent potential, force, geometry and Armijo checks.

| Quantity | Original32 cap | Continued128 cap |
| --- | ---: | ---: |
| Root-only arms converged | 59/64 | 63/64 |
| Collective arms converged | 5/64 | 46/64 |
| Pairs with all four arms converged | 2/32 | 20/32 |
| Mean root-only destination-source gap (eV) | 0.62051 | 0.59453 |
| Mean collective destination-source gap (eV) | 0.54324 | 0.50666 |
| Paired collective-minus-root gap (eV) | -0.07727 | -0.08786 |

The final paired descriptive interval is[-0.23315,0.07213] eV. It fixes four
compositions and resamples parents within them; it is not a chemical-space
generalization interval. More optimization improves convergence but does not
establish an exchange-specific collective benefit. The all32-pair denominator
remains primary. Five collective arms are still capped;14 arms stop at minimum
step. No global-minimum or intrinsic chemical-gap conclusion follows.

## Complete boundary diagnosis and finite-energy check

All14 minimum-step stops are included:9 disconnection boundaries,4 frozen-graph
changes and1 failed chemical-perception/valence assignment. The last category
was initially unsupported by the boundary-analysis script; the observed failure
is recorded in `research/evidence/mobility_boundary_analysis_recovery_v1.json`.
The script now records it and the actual contact-edge change
instead of dropping that stop. No physical calculation was repeated for this fix.

Both the raw allowed force and a distance-adjusted version are constructed from
saved forces. The latter removes outward components at observed active distance
boundaries, adds a small inward component, and uses geometry-only backtracking.
All directions and candidate coordinates were frozen BEFORE new physical
energies. This is a classical optimization diagnostic, not a learned proposal.

Raw force has5 feasible candidates and9 geometry failures. Distance-adjusted
force has14 feasible candidates. Fresh and repeated paired source evaluations
cost56 raw calls; the19 feasible candidates cost38. Total94, exactly the declared
cap. Producer46214280 and audit46214323 are COMPLETED 0:0. All47 physical states,
raw inversion pairs, graph constraints and numerical-repeat checks replay.

At the frozen1e-5 eV threshold, adjusted directions give12 decreases and2
increases. Their mean decrease is0.06921 eV, range[-0.16687,0.26568]. Raw force
gives0 resolved decreases,1 increase and4 changes within numerical tolerance;
the9 geometry failures remain in its14-stop denominator. These selected optimizer
failures are not a sampling/generalization benchmark. A lower nearby feasible
point does not by itself certify a constrained stationary point or a minimum.
The v2 summary makes that wording narrower without changing any numerical result.

## Concrete next recovery

Implement and test a constraint-aware search with actual-energy Armijo
backtracking. Geometry feasibility and positive first-order force prediction do
not guarantee a finite potential decrease: the two adverse candidates are direct
evidence. Preserve those outcomes and use them as regression cases. Keep allowed
COM/root coordinates and the actual graph/geometry validator; do not silently
weaken the target or alter charge/spin to make optimization easier.

Before additional physical queries, freeze ONE bounded recovery of all19
remaining arms (14 minimum-step stops and5 capped collective arms), retaining
all128 arms and both endpoints in the final comparison. Reuse original physical
states and preserve histories where the algorithm permits it. An explicit new
optimizer phase may reset histories, but its costs and changed algorithm must be
recorded. Do not call a small step convergence; distinguish full-force residuals,
any properly defined constrained stationarity check, and algorithmic stopping.
No next recovery protocol or physical job is frozen/submitted yet.

Keep this as a bounded diagnosis supporting a method decision. A useful
generator does not require globally minimizing every molecule. Do not train
another network solely because these optimization repairs work. Existing
whole-COM escorted paths already failed useful cost/acceptance checks; later
learned geometry transport still needs its actual density/Jacobian or complete
reversible auxiliary path. Optimization with endpoint-only MH is not valid.
Screening-only scale-up remains stopped. Full molecular advantages, complete
costs, strong physical and representative learned baselines, realistic reuse and
independent final evaluation remain required. Evaluated cohorts and722 reserved
conditions remain excluded from fitting.

Current result: `research/MOBILITY_CONTINUATION_RESULT_20260913.json`.
Summaries: `runs/mobility_relaxation_continuation_summary_v1/results.json` and
`runs/mobility_boundary_energy_summary_v2/results.json`. The working paper records
both completed outcomes in appendix N. This turn adds3,862 raw calls, bringing
the mobility/boundary diagnostic total to9,426; none are actual sampler savings.
