# Fixed broad-development entropy-refinement comparison

Full ICLR goal remains open. Extend the method to ALL eight previously frozen
development conditions, including every failed source or optimization. Reserved
conditions remain untouched. The source is the v2 deterministic global-FM64 plus
.025-A COM Gaussian protocol, with unchanged checkpoint and electronic states.

Source contract: each complete condition must match its parent-row checksum,
the frozen checkpoint/config/manifest/oracle hashes, exact charge/spin/labels,
source-model input1 eV, physical target kT=.025851999786435 eV and restraint
.1 eV/A2. Training4096 and development512 rows use the fixed independent streams
of seed9182/batch64. No reference geometry or reference energy enters training.
The objective's numerical energy offset is the median of GENERATED-source energies.
No path work, absolute density or importance ESS exists for this source.

Methods: convex species coupling, its same-neural-context affine tangent control,
and typed global linear. Two training pairs use initialization9161/9162 and
selection9141/9142, shared within each matched comparison. Neural architectures
share19365 parameters, one sweep, hidden32/radial24 and cosine lr .001 to .00001.
Typed linear keeps its stated .01 to .0001 schedule and .25 pair-weight bound.
All use1000 AdamW updates, batch16, zero weight decay, gradient clip10. No best
checkpoint selection or outcome-dependent tuning. Each arm uses17024 physical
queries:16000 training+512 source-energy replay+512 adapted development energies.
Source generation4608 queries/condition, neural inference, pretraining, failed
branches, confirmation and xTB costs remain additional, not hidden in this budget.

Engineering release: all eight conditions, all three families, one initialization,
two updates and16 evaluation rows each, using the already qualified32+32-row
engineering source. Each arm uses64 queries; full24-arm screen1536 queries.
Require cached-energy replay within .001 eV, finite gradients, complete intrinsic
Jacobian agreement on two trained geometries, inverse on up to32 rows and public
checkpoint-loader replay. Passing code checks is not sampling qualification.
The source contract has semantic-corruption tests, beyond checking file hashes.

The full24-arm screen45888794 now passes with1536 acknowledged queries and no
missing arms. Full production is released under a4-hour-per-condition cap:
condition0 can start from its completed immutable source row; conditions1--7
are queued as an array with at most two running tasks after source production.
Each task retains all six prescribed arms. Across eight conditions, the planned
adapter-query count is817152, plus36864 source queries and separately recorded
engineering/assessment/pretraining costs. These are planned budgets, not results.

Production runs use one condition per bounded allocation, with all six trained
arms and every failure retained. A completed selected source row may be used
while later source conditions are still running; the complete eight-condition
denominator must remain in final aggregation. Report raw and per-internal-DOF
paired relative-KL changes, energy, log volume, all unweighted geometry/diversity
diagnostics and independent xTB with failure denominators. Row SEM is conditional
on each trained model; do not pool matched seeds as independent data. Fresh
confirmation and strong external equivariant-flow/HMC controls remain necessary.

## Producer timeout and bounded RPC repair

The original full producer45885669 requested4096 structures through one synchronous
CPU oracle RPC, whose response limit is60 seconds. Condition0 passed deterministic
FM replay but timed out before the bulk energy reply; condition1 was still active.
After identifying this execution defect the job was cancelled at14m12s. Neither
condition had completed sample artifacts. Both partial records remain untouched.
Zero acknowledged attempts in the failed report does NOT mean zero physical
compute. Condition0 submitted4096; condition1's submitted count was not retained.
The overall attempted-query bound is0--8192 for the two started training streams,
with zero qualified outputs and actual scheduler time retained separately.

The replacement uses at most32 structures per RPC, retaining the worker's internal
physical batch size16. Source coordinates are written before each physical call,
and separate scored chunk files preserve completed energy/force results. Each
batch's acknowledged/requested counters are recorded. This changes orchestration,
not the frozen source or target. Bounded-RPC unit tests cover values/order,
successful accounting and unacknowledged timeout cost. A real first/last-condition
128-query check must pass before releasing the replacement full producer.

The real check45888793 passed: both64-row panels replay cached energies exactly;
32-structure RPCs take1.14--4.15 seconds. Replacement producer45889306 is running
with the same frozen source law and target and persistent scored chunks.
