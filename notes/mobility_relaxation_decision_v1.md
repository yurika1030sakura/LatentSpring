# Paired mobility result and next decision

The frozen 32-pair, 128-arm diagnostic is complete. Producer46212367 and
audit46212970 are COMPLETED 0:0, with 5,564 new raw physical calls versus the
8,448 cap. Four full replays and independent force, geometry, COM, passive-
coordinate and Armijo checks cover all 2,654 queried optimization trials.
`research/evidence/mobility_relaxation_summary_audit_v1.json` separately checks
all paired summaries against compact producer results. The original traces
and every early stop remain unchanged.

Eight FIT parents in each of four compositions are selected by the frozen hash
rule; one supported pair per parent is selected without energy or acceptance
outcomes. Thirty pairs change canonical connectivity. Both source and destination
retain their own perceived bond-order graph and electronic state. These are
optimization trajectories, not physical dynamics or a new transition proposal.

| Quantity | Mean eV | Descriptive parent interval |
| --- | ---: | --- |
| Initial destination minus source | 1.30174 | [0.97871, 1.68233] |
| Root-only bounded gap | 0.62051 | [0.39669, 0.85600] |
| Collective bounded gap | 0.54324 | [0.32640, 0.77009] |
| Collective minus root-only gap | -0.07727 | [-0.22250, 0.08523] |

Collective motion lowers destinations an additional 1.10282 eV but also lowers
sources an additional 1.02555 eV. The destination-only change would overstate the
evidence that collective geometry particularly helps chemical exchange. This
paired contrast does not establish such an effect. Intervals resample parents
within four fixed compositions; they do not establish chemical-space transfer.

The unresolved issue is convergence. Root-only has 59/64 converged, four capped,
and one minimum-step stop. Collective has 5/64 converged, 49 capped and ten
minimum-step stops. Only 2/32 complete pairs converge. The best feasible observed
points are not global minima, and most collective points are not even qualified
local stationary points. A positive remaining gap cannot therefore identify an
intrinsic chemical energy difference. Equal caps also do not mean equal actual
cost: roots use 1,638 optimization raw calls, collective 3,670, and shared initial/
repeatability checks 256. At caps8/16/32 the mean paired contrasts are
-0.14340/-0.11413/-0.07727 eV; they do not establish an asymptotic trend.

Inspection of the saved final rejected geometry events finds nine minimum-step
stops reporting the connected/no-overlap domain, and two reporting frozen-graph
change. Those are rejection categories, not identification of the active atom
pairs or constrained stationarity. More iterations alone will not resolve a
boundary stop. The raw oracle numerical repeats are small: maximum energy
repeat discrepancy 7.95e-7 eV and force component discrepancy 1.71e-6 eV/Angstrom.
These pass the frozen tolerances; numerical consistency is not physical accuracy.

Next inspect the actual limiting distances/bonds and force components at those
saved boundary events without new oracle calls. Then, if extending the diagnostic,
freeze a separate bounded continuation of every budget-exhausted arm, preserving
optimizer state and replaying cached prefixes. Retain all converged and blocked
arms in the final denominator. Do not select only favorable pairs, silently
relax support, restart from scratch or treat a 32-query cap as a scientific
rejection deadline. No continuation protocol or job is frozen/submitted yet.

Do not construct another neural head from this result alone. Existing whole-COM
escorted paths have already failed useful acceptance/cost tests. Any new learned
geometry transport must carry its actual density/Jacobian or the complete
reversible auxiliary path. Uncontrolled relaxation with endpoint-only MH is
incorrect. The FM initializer, OMol25 primary data, max_atoms200, bond-free
supervision and declared physical target remain unchanged. Evaluated cohorts
and722 reserved conditions remain excluded from fitting.

Evidence: `research/MOBILITY_RELAXATION_RESULT_20260913.json` and
`runs/mobility_relaxation_summary_v1/results.json`. The paper records this as an
inconclusive diagnostic in appendix N. It is not a useful AI-method result or
an equilibrium certificate. The full ICLR goal remains active.
