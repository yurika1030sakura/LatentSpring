# Claude continuation — September 12, 2026

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `CLAUDE.md`, `research/NEXT.md`,
`research/STATUS.md` and `research/ACCEPTED_UTILITY_STATE_20260912.json`.
Keep the environments separate and do not write to the home checkout.

The bounded geometric accepted-utility learner is implemented, trained with two
seeds and audited. Differentiable observed joint densities include the actual
reverse contexts, both orders and all normalizers. Production draws remain
no-grad. Both seeds improve the internal offline utility-per-call proxy by about
2.7%, with stable importance weights. This does not establish a useful sampler.

The fresh-proposal follow-up is COMPLETE: 72 fixed internal source states,
3,456 attempts, 6,708 raw calls and 3,354 independent MH checks. Actual utility
differences are near zero and both parent intervals span zero. Small effects
remain possible; no fresh-proposal gain or complete-chain benefit is established.
Do not scale this frozen geometry-only recipe. Jobs 46186496, 46186856, 46186968,
46189583 and 46189707 are terminal; re-query Slurm before action.

Next implement the bounded conditional action/geometry comparison in
`notes/bounded_action_geometry_candidate_v1.md`. Read the old
`cfm_mol/chemical_policy.py` and `notes/chemical_policy_decision_v2.md` first:
the earlier selector failed its sampling/cost gate. Keep the move-family schedule
fixed, compute full forward/reverse action probabilities, and match total
likelihood-ratio bounds across ablations. This is a hypothesis, not established
novelty. Freeze the protocol before fitting. Initial work needs no new oracle.

Use only the existing 36 FIT parents; retain the frozen 12-parent internal
selection split. Do not fit on fresh follow-up outcomes, either evaluated
molecular cohort, or the 722 reserved conditions. Data construction costs 18,110
raw calls; do not inherit the scalar learner's different overhead. Preserve the
scalar-chain failure and all earlier strong-control negatives.

The manuscript includes both offline and fresh-proposal results:
`runs/verification/accepted_utility_20260912/main.pdf` (8 main pages, 20 total).
Formatting passes; scientific submission readiness is false. Continue the full
ICLR goal; authors and actual submission remain with the user.

Prior handoff: `notes/archive/claude_handoff_through_scalar_chain_20260912.md`.
