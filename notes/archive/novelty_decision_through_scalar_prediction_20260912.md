# Current contribution decision — September 12, 2026

There is a concrete learned-method candidate, with a positive internal prediction
signal and an independently checked complete proposal implementation. There is
not yet a demonstrated learned molecular sampling advantage or an established
ICLR contribution. Read `research/CONDITIONAL_NONLOCAL_STATE_20260912.json`.

The new masked conditional scalar model predicts nonlocal placement work and
supplies scores on analytically feasible circle arcs. Both work-only and work-
plus-force objectives improve held-out work prediction across two seeds. The
added force loss has no demonstrated decisive advantage over work alone; do not
turn that extra term into the paper's novelty claim. A useful learned coupling
of molecular connectivity changes,3D placement and tractable support-restricted
conditional probabilities is the intended contribution.

The relevant ingredients already have precedents: classical configurational-bias
regrowth, self-learning MC with effective neural Hamiltonians, learned probabilistic
decoders, and geodesic sphere sampling. The primary sources and exact boundaries
are in `notes/nonlocal_arc_prior_art_20260912.md`. A cheap learned potential plus
MH correction alone is not novel. Differences from recent graph-energy models
and learned molecular MCMC need representative empirical comparisons as well as
clear task/target distinctions. A failed repurposed EACF refiner is not a native
EACF benchmark.

The previous vector learner still loses its demonstrated advantage to stronger
physical concentration64/400 controls. Its original GFN2 result covers only its
old concentration10 comparison. Keep all adverse transfer, initialization and
cost results. The new scalar predictor does not retroactively repair them.

Next evaluate both scalar objectives in complete chains on48 internally withheld
parents, at exact accounted data costs and same-inference readouts against physical
site arcs. Internal prediction improvement cannot substitute for that test.
After a useful signal, include stronger biased-regrowth/same-score local controls
and appropriate learned baselines on a frozen broader scope. Do not imply that
terminal H/halogen exchanges cover all molecular connectivity changes.

Keep both zero-support training conditions. One has empty connected support under
the pinned builder; that is target incompatibility, not chemical impossibility.
Do not fit on prior evaluated-six-composition coordinates or query722 reserved
outcomes before final method/scope freeze. The user's standard does not require
universal perfection or a new physical law, but it does require correct and useful
evidence for the actual claims. The project remains scientifically unready.

Prior decision: `notes/archive/novelty_decision_through_geodesic_physics_20260912.md`.
