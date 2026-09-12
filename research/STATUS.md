# Current research status — September 12, 2026

Authoritative checkpoint: `research/ACTION_GEOMETRY_STATE_20260912.json`.
The ICLR goal remains active and scientifically unachieved. The completed
conditional action/geometry comparison has no established combined advantage.

## Six-arm comparison completed

Three variants (action-only, geometry-only, joint), two seeds, 300 fixed steps,
and the same 36 FIT / 12 internal-selection parents use identical data,
minibatch draws, optimizer and complete likelihood-ratio bound exp(2). The
move-family schedule is fixed. Active parameter counts/runtime are reported
separately. All 1,671 attempts, including 74 failures, remain. Training and
audits use no new physical queries; inherited data construction costs 18,110.

Internal utility-rate differences versus physical initialization, in
1e-5 eV per expected raw call, with descriptive 95% parent-bootstrap intervals:

| Variant | Seed 0 | Seed 1 |
| --- | --- | --- |
| Action only | -6.27 [-17.11, 3.62] | -5.60 [-16.28, 4.07] |
| Geometry only | +8.82 [3.55, 15.61] | +8.76 [3.12, 15.59] |
| Joint | -0.68 [-11.02, 9.54] | -0.59 [-9.12, 8.85] |

The physical rate is 0.003225565 eV per expected raw call. Action-only and joint
point changes are -1.94% / -1.74% and -0.21% / -0.18%; their intervals span zero.
Joint improves on action-only in this proxy but does not establish improvement
on geometry alone or physics. The allocation of the common bound limits this
conclusion; it is not a theorem excluding every joint policy. Intervals are
internal, descriptive, fixed-composition and not multiplicity-adjusted.

All final/baseline metrics replay. Six audits check 576 independent full
forward/reverse densities (maximum error 4.98e-14) and eight trained-head finite
differences (maximum error 3.17e-10). Geometry-only parameters reproduce previous
models within 7.8e-16; its roughly 2.7% offline gain is unchanged. Old and new
intervals use different fixed bootstrap draws, so finite-bootstrap endpoints
differ slightly. Summary: `runs/action_geometry_summary_v1/results.json`.
Training 46192301 and audit 46192339 are complete with exit code zero.

## Existing molecular negatives remain

The preceding geometry-only fresh-proposal check (3,456 attempts, 6,708 calls)
does not establish actual utility gain. Its intervals still allow small effects;
the population differs from the full offline proxy. The scalar work and
work-plus-force models fail the 48-parent complete-chain comparison after data
costs, and neither establishes same-inference energy benefit. The old vector
learner's stronger-control and transfer failures are unchanged. See
`research/ACCEPTED_UTILITY_STATE_20260912.json` and
`research/CONDITIONAL_CHAIN_RESULT_20260912.json`. Do not scale those weights.

## FIT-only query-cost diagnosis

The 36 FIT parents contribute 1,242 joint attempts, 1,186 scored, costing 2,372
calls out of 9,216 complete-trajectory calls. Balanced expected acceptance among
scored joint proposals is 5.35%. Expected calls associated with rejected joint
moves total 2,244.0, or 24.35% of recorded trajectory calls. This describes this
move family and fixed empirical prefix; it does not predict achievable savings
or a changed chain's behavior. Preparation and learning costs are additional.
An oracle-informed screening diagnostic uses the unseen candidate energy and
is explicitly not deployable. Actual saved calls are zero. Separate oracle and
geometry timings are unavailable, so no wall-time claim follows.

Read `notes/delayed_acceptance_prior_art_20260912.md` before the next bounded
screening investigation. Classical, neural and structural-move delayed acceptance
already exist. A corrected cheap screen may be worth testing; generic screening
cannot supply novelty by renaming it. No new screen or protocol is implemented
or frozen yet. Leave all evaluated outcomes and the 722 reserved conditions out
of fitting. The development manuscript includes the completed negative result;
formatting completion does not establish scientific readiness.

Historical status: `notes/archive/status_through_action_geometry_running_20260912.md`.

Verified development PDF: `runs/verification/action_geometry_completed_20260912/main.pdf`
(9 main pages, 21 total). Build record:
`research/evidence/action_geometry_completed_build_20260912.json`. Scientific
submission readiness remains false.
