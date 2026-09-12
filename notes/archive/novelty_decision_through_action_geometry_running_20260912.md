# Current contribution decision — September 12, 2026

The bounded geometric accepted-utility learner is implemented and audited, but
has not established a useful sampling contribution. Both seeds improve an
internal offline proxy by about 2.7%; fresh actual-oracle proposal checks have
near-zero point differences and intervals spanning zero. Do not scale these
weights or call the offline result a molecular generation gain.
Read `research/ACCEPTED_UTILITY_STATE_20260912.json`.

The intended contribution remains useful learning of coupled molecular graph
changes and 3D placements with computable proposal probabilities and realistic
costs. Exact MH, biased regrowth, neural surrogate energies, bounded logits and
accepted-movement objectives have prior art. The importance-flow identity and
bounded-score ratio bounds are elementary mathematics, not new general theorems.
See `notes/nonlocal_arc_prior_art_20260912.md`.

Conditional action selection together with feasible placement is now implemented;
the frozen six-arm comparison and independent audits are running. Read
`research/ACTION_GEOMETRY_STATE_20260912.json`. No advantage follows yet.
This differs from the earlier failed selector only if it keeps the move-family
schedule fixed, uses audited regrowth geometry, trains signed cost-adjusted
accepted utility and includes the full inverse-action probability. Read
`notes/chemical_policy_decision_v2.md` before implementation. Additional action
capacity alone does not establish novelty or resolve a geometry bottleneck.

A matched action-only / geometry-only / joint comparison can test whether the
coupling is useful. Equal likelihood-ratio bounds, data and optimizer
budgets are frozen; preserve every failed attempt in denominators. No
fitting on fresh validation outcomes, either evaluated molecular cohort or the
722 reserved conditions. A positive empirical one-step proxy still needs fresh
sampling validation, strong physical and representative learned baselines,
realistic reuse costs and independent final evaluation.

The scalar-chain failure and old vector learner's stronger-control/transfer
failures remain unchanged. Lower finite-budget potential is not an equilibrium
distribution result. A failed repurposed EACF refiner cannot substitute for native
EACF performance. The user does not require perfection or a new physical law;
the proposed contribution must nevertheless be correct, distinctive and useful.
The ICLR goal remains active and the paper scientifically unready.

Prior decision: `notes/archive/novelty_decision_through_scalar_chain_20260912.md`.
