# Current paper story — September 12, 2026

The research question is how to propose useful molecular connectivity changes
with compatible3D geometry and computable reverse probabilities. An OMol25-trained
flow-matching initializer supplies coordinates with unknown finite-step density;
it remains frozen. A separate corrector proposes coupled connectivity, radius
and angular updates, then uses the full MH ratio for the declared restrained,
inversion-averaged eSEN target on algorithmic molecular support.

The newest reconstruction computes feasible angular arcs and their normalized
proposal probabilities. It accounts for both circle orientations and both root
orders in the complete joint density. This substantially improves proposal support
in the audited training screen. Generic geometry, MH, force fitting and geodesic
walks are established ingredients; their correctness alone is not AI novelty.

The intended AI contribution is a useful conditional energy/proposal model that
learns how to move across these feasible regions at lower total physical cost.
That contribution remains unproved. The previous normalized-site learner loses
its demonstrated advantage to stronger physical controls. New local force/work
fits repair a local identification problem but do not provide a useful global
teacher: even on valid arcs, a uniform proposal makes more single-step expected
energy progress. A full-chain physical comparison is now running to guide the
next model design. No new neural weights are claimed as a successful repair.

The current manuscript `paper/angular_working.tex` reports the earlier candidate
and its adverse controls; it has not incorporated this new arc round. Lower
finite-cost energy and validator acceptance do not establish an equilibrium energy
distribution or quantum stability. Useful held-out performance against strong
physical and representative learned baselines is still required on a declared scope.

See `research/GEODESIC_RECONSTRUCTION_STATE_20260912.json`, `research/NOVELTY_DECISION.md`
and `research/CLAIM_AND_BENCHMARK_SCOPE.md`. The project remains scientifically
unready for submission. Earlier story:
`notes/archive/paper_story_current_through_sharp_training_20260912.md`.
