# Current research status — September 12, 2026

Authoritative checkpoint: `research/SOURCE_FORCE_SCREEN_STATE_20260912.json`.
The full ICLR goal remains active and scientifically unachieved. The completed
source-force learner does not establish a useful advantage over cheap controls.

## General gates and completed experiment

Forward gates use the current state's cached even force, proposed geometry,
perceived graphs, electronic conditions and known proposal ratio. Candidate
force is used only after a passing gate and a paid query. The second log ratio
is R + log(g_reverse) - log(g_forward), not an assumed reciprocal R-s. Tests
cover molecular/force symmetries, information order, zero-gate exact RNG/query
replay, and visible error after an invalid paid reverse gate. Nineteen combined
regressions pass; independent FIT preview agrees to2.23e-16.

A five-coefficient linear model and paired-geometry/force GNN each train for300
steps and two seeds. Physical, source-work and zero controls are fixed. Each
learner's constant-thinning control is calibrated only from FIT query use.
The unchanged data contain36 FIT /12 internal-selection parents,1,671 attempts,
74 failures, and verified original force/electronic provenance. No new oracle
calls are used; inherited data construction still costs18,110 raw calls.

The objective/evaluation now includes initial and nonjoint work/cost constants
from the original complete prefixes. The72 FIT prefixes cost128 calls each and
have mean expected work0.4947591 eV, giving the fixed rate0.0038653057 eV/call.
Joint contributions are replaced at recorded states; the changed trajectories
are not replayed. All96 original prefixes and1,671 joints are joined and checked.

## Internal whole-prefix results

Expected work per expected raw call, eV, on12 internal parents and24 prefixes:

| Model | Seed0 | Seed1 |
| --- | --- | --- |
| No screen | 0.004167470 | same fixed control |
| Fixed physical | 0.004733094 | same fixed control |
| Source work | 0.004522226 | same fixed control |
| Linear | 0.004224067 | 0.004263457 |
| Neural | 0.004420138 | 0.004258544 |

Neural gains versus no screen are6.06% /2.19%; fixed physical gives13.57% and
source work8.51%. Neural-minus-physical differences are-0.0003130 /-0.0004745,
with95% parent intervals[-0.0005384,-0.0001026] and[-0.0008875,-0.0001086]. Both
favor the fixed physical control. Only the first neural seed beats its matched
thinning baseline; neither establishes an advantage over the linear model.
Intervals fix four compositions, retain both trajectories per parent and are
not multiplicity-adjusted. Summary: `runs/source_force_screen_summary_v1/results.json`.

All final/control metrics replay. Independent calculations cover6,388 supported
pair cases with both gates and balance, max discrepancy2.09e-14. Six sensitive
trained-head gradients agree with FD to8.51e-10. Jobs46199581/46199747 completed
with exit code zero. This is still a fixed-source expectation, not achieved
query savings, changed-chain endpoints, equilibrium or a wall-time speedup.
The earlier93% physical-screen gain used conditional joint-query rates; the
13.57% value uses whole-prefix denominators, so those numbers are compatible.

## Next bounded learning question

`runs/gate_teacher_diagnostic_v1/results.json` uses only FIT outcomes. Neural
seed0 retains98.95% of expected accepted moves but costs7.26 times the oracle-
informed minimal gate; seed1 retains75.67% and costs5.71 times that teacher.
Physical screening retains51.20% and costs3.00 times it. These are diagnostics,
not evidence for a different primary success metric or a mixing comparison.
The oracle-informed gate uses unknown candidate energy and is not deployable.

The bounded oracle targets and a simple underprediction/retention relation are
implemented in `cfm_mol/gate_teacher.py`; two tests pass. A next candidate tests
dense target pretraining followed by utility refinement against a matched total-
step direct-utility control, with both linear/neural and two seeds. No such
protocol or model is frozen/fitted yet. See `notes/gate_distillation_candidate_v1.md`.
No new physical queries are needed initially. Keep all selection/fresh/prior
molecular-evaluation and722 reserved outcomes out of fitting. Prior action/geometry,
scalar-chain, strong-control and transfer negatives remain unchanged.

Historical status: `notes/archive/status_through_delayed_screen_20260912.md`.

Verified development PDF: `runs/verification/source_force_screen_completed_20260912/main.pdf`
(9 main pages,23 total). Build evidence:
`research/evidence/source_force_screen_completed_build_20260912.json`. Scientific
submission readiness remains false.
