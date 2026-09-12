# Current contribution decision — September 12, 2026

The bounded delayed-screen comparison is complete and audited. Linear and
neural models improve a recorded-pair query-rate proxy over no screening, but
the fixed physical screen has a higher point estimate. Neural-versus-linear
intervals span zero. This does not establish an added neural contribution or
actual sampling gain. Read `research/DELAYED_SCREEN_STATE_20260912.json`.

Delayed acceptance, learned surrogates, derivative-informed acceleration and
screening of structural moves have close prior art; see
`notes/delayed_acceptance_prior_art_20260912.md`. The exact correction is required
for validity, not claimed as a new theorem. Large per-joint-query changes do not
establish whole-kernel speedups, and no actual calls have been saved in this
recorded-pair experiment. Do not scale these neural weights.

The next possible input is a cached source force. It is already part of the
physical sampler and may support a work estimate before querying a candidate.
This is a first-order approximation with known limitations, not exact finite
work. A source-only gate is generally nonreciprocal; its reverse probability
must be computed after querying the candidate and included in the correction.
The force-augmented data are audited, but no such model or frozen experiment
exists yet. See `notes/cached_force_screen_candidate_v1.md`.

Any contribution must demonstrate useful molecular learning against fixed
physical and simple learned controls, with actual chain/query/wall-time/data
costs and independent final evaluation. Include a thinning/frequency control
when interpreting screened-chain gains. No fitting on internal selection,
fresh follow-up outcomes, either previous molecular evaluation cohort or the
722 reserved conditions. The old action/geometry, scalar, vector and stronger-
control negatives remain; standard MH/force principles and a new neural head
alone do not satisfy novelty. The user requires a useful defensible method,
not perfection or a new physical law. The full ICLR goal is active and the
paper scientifically unready.

Prior decision: `notes/archive/novelty_decision_through_action_geometry_complete_20260912.md`.
