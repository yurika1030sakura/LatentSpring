# Decision after matched gate training and larger-cohort validation

The matched800-step gate experiment is complete. Pretraining loses to direct
utility on the original12 selection parents. On the larger reused48-parent,
six-composition cohort, the pretrained neural models have small positive point
changes against the physical screen, but negative point changes against no
screening; all relevant intervals span zero. None of the tested screens
establishes a gain over no screening. These are still fixed-prefix expectations,
not actual screened chains or saved queries. Stop scaling or tuning the tested
screening-only recipes. Do not choose a favorable control or subset to claim
an AI contribution.

The next useful question concerns proposal geometry and intrinsic chemical
energy differences. The current joint site-arc update regenerates two terminal
positions while preserving passive relative coordinates. A better acceptance
filter cannot create endpoints that this proposal fails to reach. This is a
hypothesis about the limitation, not its proven cause.

## Inspect existing collective/path evidence first

Read `notes/nonequilibrium_chemical_candidate.md`, `cfm_mol/escorted_exchange.py`,
`cfm_mol/chemical_path_guide.py` and their v1/v2/v3 audits before constructing any
new relaxation move. Whole-COM pre/post Langevin paths already exist. The v1/v2
physical pilots accepted only4/64 paths per replica even after the graph guide
raised endpoint support to64/64 and63/64. Do not rediscover that path, discard its
negative result or assume that collective motion alone is novel or sufficient.
Uncontrolled relaxation followed by endpoint-only MH would omit proposal/path
probabilities and is not an acceptable replacement.

## A bounded physical diagnostic to make the next action concrete

First audit the cached FIT proposal forces and the existing path failures. Large
forces on frozen anchors alone do not prove a bottleneck: a strained bond creates
reaction forces on both atoms and leaf motion may relieve both. A discriminating
experiment would compare root-only and collective relaxation for BOTH source and
exchanged destination, at matched bounded oracle budgets and fixed perceived
connectivity. The paired source control separates generic relaxation of an
imperfect initializer from geometry costs caused specifically by the exchange.

Select only FIT pairs by a frozen rule, retain every failed/blocked relaxation,
and report physical potential changes, support, residual forces and actual raw
calls. Distinguish canonical connectivity changes and same-connectivity moves;
label permutations or rigid motions are not new physical samples. Compare the
source-to-destination gap after both mobility choices, rather than only showing
that more degrees of freedom lower a destination's energy. Finite optimization
is a teacher-feasibility diagnostic, not an equilibrium or sampling result.
That prospective protocol was subsequently frozen and completed; see the
section below and `notes/mobility_relaxation_decision_v1.md` for its audited outcome.

If collective relaxation provides specific evidence, it can motivate a learned
proposal that transports compatible geometry with an explicit density or fully
accounted reversible auxiliary path. If intrinsic chemical energy gaps dominate,
address action/target coverage instead. Do not add another neural head without
this distinction. Preserve the FM initializer, OMol25 primary data, electronic
states, max_atoms200, bond-free supervision and all prior failures. Keep every
evaluated cohort and722 reserved outcomes out of fitting. The full ICLR goal is
active; competitive AI benefit, real sampling gains and submission readiness
remain unestablished.

## Frozen paired mobility diagnostic

The L-BFGS/Armijo diagnostic is implemented in `cfm_mol/mobility_relaxation.py`
and `scripts/research/mobility_relaxation_pilot.py`. Roots-only and collective
coordinates use orthonormal COM charts; accepted/trial graphs must retain the
original endpoint bond orders. This is optimization, not a transition proposal.
Analytic constrained minima, geometry backtracking, real-graph query replay and
independent force/Armijo checks pass four tests.

`research/evidence/mobility_relaxation_protocol_v1.json` freezes32 pairs from32
FIT parents (eight per composition), selected by hashes without energy or
acceptance outcomes. Thirty change canonical connectivity. Both source and
destination are optimized with both mobility choices, with32 candidate energy
queries per arm. Early convergence and line-search blocking remain explicit.
The source/destination comparison must use both mobility choices; more endpoint
relaxation alone is not evidence of a proposal advantage.

Fresh initial energies/forces are shared across mobility arms and repeated to
check numerical consistency. The oracle uses singleton internal batches;
initial differences from archived values are retained. The total upper bound is
8,448 raw calls (8,192 optimization plus256 initialization/repeatability), with
both inversion orientations counted. No global-minimum, equilibrium, sampler or
AI benefit follows from this diagnostic. Read NEXT for live job handles.

## Completed result

Producer46212367 and audit46212970 are complete. The paired gap contrast is
-0.07727 eV with an interval spanning zero; collective convergence is only5/64.
Do not infer intrinsic chemical minima or useful transport from this bounded
diagnostic. Read `research/MOBILITY_RELAXATION_RESULT_20260913.json` and
`notes/mobility_relaxation_decision_v1.md` before a continuation.
