# Next research actions — September 13, 2026 UTC

Read `research/MOBILITY_CONTINUATION_RESULT_20260913.json` and
`notes/mobility_relaxation_continuation_decision_v1.md`. Work in laboratory storage;
never write home. The full ICLR objective is active and scientifically unachieved.

1. The128-cap continuation and all audits are COMPLETE. All53 previously capped
   arms were continued with cached-prefix replay; the other75 arms were retained.
   Added cost3,768 raw calls, cumulative9,332. Root/collective convergence is now
   63/64 and46/64, respectively;20/32 full pairs converge. Fourteen arms stop at
   minimum step and five remain capped. All4,538 optimization trials are audited.
2. The mean paired collective-minus-root gap change is -0.08786 eV with interval
   [-0.23315,0.07213]. This does not establish an exchange-specific geometry gain.
   Keep the all32-pair denominator and both source/destination controls. Shorter
   results, actual costs, failed cases and unconverged residuals remain.
3. All14 boundary stops and both raw/distance-adjusted force candidates were
   frozen and checked with94 new raw calls. Four independent audits are complete.
   Distance-adjusted directions give12 resolved decreases and2 increases. Raw
   force gives0 resolved decreases,1 increase,4 unresolved changes and9 geometry
   failures. The numerical threshold was frozen at1e-5 eV. All28 rows remain.
4. Implement/test a constraint-aware optimizer recovery with actual-energy
   Armijo backtracking. The finite candidates show why geometric feasibility and
   force prediction alone are insufficient. Before new queries, freeze ONE bounded
   recovery of all19 remaining arms, preserving histories/caches where applicable,
   both mobility choices, both endpoints and all128 arms in the denominator. Keep
   graph, electronic state and the declared physical target unchanged. No recovery
   protocol or physical job is yet frozen/submitted.
5. Explicitly distinguish full-force convergence, constrained stationarity and
   an algorithmic stop. Do not declare success merely because steps become small
   at the graph/domain boundary. Keep the paired diagnostic bounded; a useful
   generative method does not require finding global minima for every molecule.
6. Optimization is teacher feasibility. Existing whole-COM escorted paths failed
   their useful acceptance/cost test. Any later learned transport requires its
   actual density/Jacobian or a complete reversible auxiliary path. These classical
   repairs do not themselves establish useful AI novelty or a valid new sampler.
7. Screening-only scaling stays stopped. All prior scalar, geometry, action,
   strong-control and transfer negatives remain. Keep evaluated cohorts and722
   reserved conditions out of fitting. Real molecular gains, complete costs,
   strong physical and representative learned baselines, realistic reuse and
   independent final evaluation remain requirements of the full goal.

All new jobs46213860,46213924,46214280,46214323 are terminal; re-query Slurm before
recovery. Summaries:
`runs/mobility_relaxation_continuation_summary_v1/results.json` and
`runs/mobility_boundary_energy_summary_v2/results.json`.
Verified PDF: `runs/verification/mobility_continuation_boundary_20260913/main.pdf`
(9 main pages,26 total). Scientific submission readiness is false.
Prior NEXT: `notes/archive/next_through_mobility32_20260913.md`.
