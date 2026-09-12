# Current research status — September 12, 2026

Authoritative checkpoint: `research/DELAYED_SCREEN_STATE_20260912.json`.
The full ICLR goal remains active and scientifically unachieved. Delayed
screening has a positive internal query-rate proxy, but no learned advantage
over suitable cheap controls or actual molecular efficiency gain is established.

## Completed bounded delayed-screen comparison

The physical site-arc proposal and move-family schedule are fixed. Zero and
physical screens are fixed controls; linear and neural screens each have two
300-step models trained only on 36 FIT parents. The same 12 internal-selection
parents, 1,671 attempts and 74 failed attempts remain. Antisymmetric factors are
bounded by log(16); the exact second acceptance step corrects the cheap screen.
A zero screen reproduces original RNG streams, queries and trajectories exactly.
The screen sees no candidate oracle values before the first decision.

Internal expected utility per raw call (eV):

| Screen | Seed 0 | Seed 1 |
| --- | --- | --- |
| No screen | 0.003225565 | same fixed control |
| Fixed physical | 0.006225109 | same fixed control |
| Linear | 0.004225544 | 0.004607301 |
| Neural | 0.004225887 | 0.004607253 |

The learned gains over no screening are about 31% / 43%; their descriptive
parent intervals exclude zero. The fixed physical control's point gain is 93%,
with difference interval [0.0003541, 0.0059175] eV per expected call. Learned-minus-
physical intervals span zero and do not establish a learned advantage. Neural-
minus-linear differences are +3.42e-7 / -4.81e-8, with intervals spanning zero.
The extra neural model has no established value in this comparison.

Learned screens retain about 94.7% / 95.4% of signed utility while expected joint
query cost falls to 72.3% / 66.8% of no screening. These are recorded-pair
expectations, not actual saved calls or screened-chain results. Expected rejected
joint calls occupy only 24.35% of the preceding complete FIT trajectory budget;
large conditional query-rate changes cannot be called whole-sampler speedups.
Preparation and learning costs are additional. All intervals fix four
compositions, resample parents with both trajectories and are not multiplicity-
adjusted. Summary: `runs/delayed_screen_summary_v2/results.json`; v1 is retained.

Every final/fixed-control metric replays. Independent NumPy checks cover 1,597
supported pairs per model, including both orientations and balance: 6,388 pair
cases, maximum discrepancy 3.09e-14. Six sensitive trained-head finite differences
have maximum error 4.93e-10. Training 46196124 and audit 46196212 are complete.
No new physical queries or actual query savings were produced by this experiment.
Data construction costs remain 18,110 raw calls.

## Implementation and next information source

Actual screened dispatch distinguishes geometry validity from scored proposals;
first-stage rejects do not query the candidate potential. Mathematical/sampler
checks passed 22 tests, followed by 17 screen/oracle/joint regressions. A fake
worker exposed a buffered-stdout timeout in the tensor oracle. Reusing the
NumPy oracle byte-buffer reader fixes it; numeric values and counters pass.
Energy RPC time is recorded separately, including failed calls. Old runs retain
original timing; no historical failure is inferred from the fake-worker test.

`runs/screen_force_pairs_v1` now attaches already-paid source forces to all 1,671
attempts and candidate forces to 1,597 scored pairs. Original split, charge/spin,
positions and trace hashes are preserved. All 3,233 distinct used force states
and projected scores reconstruct exactly from raw/inverted outputs. No new
queries or fitting were used. Candidate force must remain unavailable to the
forward pre-query screen. A source-force candidate needs a general nonsymmetric
screen correction; it is not yet implemented or frozen. Read
`notes/cached_force_screen_candidate_v1.md` and NEXT.

The action/geometry null result, scalar-chain failure, old vector/strong-control
and transfer negatives, and geometric fresh-proposal null result remain unchanged.
Their complete checkpoints are `research/ACTION_GEOMETRY_STATE_20260912.json`,
`research/ACCEPTED_UTILITY_STATE_20260912.json` and
`research/CONDITIONAL_CHAIN_RESULT_20260912.json`. Do not scale those weights.
Keep every evaluated cohort and the 722 reserved outcomes out of fitting.

Historical status: `notes/archive/status_through_action_geometry_complete_20260912.md`.

Verified development PDF: `runs/verification/delayed_screen_completed_20260912/main.pdf`
(9 main pages, 22 total). Build record:
`research/evidence/delayed_screen_completed_build_20260912.json`. This does not
establish scientific submission readiness.
