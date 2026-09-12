# Candidate after bounded geometric utility learning

The geometry-only accepted-utility learner improves the internal offline proxy by
about2.7%, but fresh actual proposals have point differences near zero with parent
intervals spanning zero. This does not prove zero benefit or refute a small gain;
it does not establish an effect worth large-scale molecular production. Keep both
results and the earlier scalar-chain failure.

A concrete next investigation is to learn WHICH chemical exchange to attempt
alongside its geometric placement. The current joint kernel still chooses legal
exchanges uniformly. Its bounded geometry correction cannot change that decision.
This is a hypothesis about an additional source of useful learning, not a proven
explanation of the current result.

Do not rediscover or silently restart the old selector. Read
`cfm_mol/chemical_policy.py`, `notes/chemical_policy.md` and
`notes/chemical_policy_decision_v2.md`. That selector used deterministic coordinate
exchange maps and a different, symmetric movement reward; it failed its sampling/
cost comparison. Its geometry bottleneck and varying local-family frequency are
explicit negative evidence. The new investigation must keep the existing fixed
local/rotation/joint schedule and use the already audited regrowth geometry.

## Required construction

- Reuse the existing invariant action-feature encoder, preserving its old API and
  results. A new bounded CONDITIONAL action policy should normalize only over
  actual legal joint exchanges. Do not learn or omit the local-family probability
  while claiming the schedule is fixed. Prefer continuous atomic descriptors to
  previously untrained element embedding rows, with explicit symmetry tests.
- The complete learned forward density is
  log p_theta(a|x) + log q_theta(y|x,a), and reverse density uses the newly computed
  legal set at y and the inverse action. The behavioral forward density is
  log q_b(y|x,a) - log N_forward. With these FULL densities, set the separate
  action-log-ratio argument to zero in the accepted-utility identity. Do not count
  N_forward/N_reverse twice. Test against an exact finite-state model with unequal
  action counts and actual-map gradients.
- If action logits have magnitude bound B_a, their probability ratio to uniform
  is bounded by exp(2 B_a). With two-root geometric score bound B_g, the complete
  ratio is bounded by exp(2 B_a+4 B_g). This is an elementary likelihood-ratio
  bound, not a new sampling theorem. Compare action-only, geometry-only and joint
  variants with the same total bound, physical initialization and data/optimizer
  budget so that a larger trust region does not explain a putative gain.
- Keep every unsupported attempt in empirical denominators. Learn the signed,
  actual-MH, cost-adjusted potential utility already defined in the accepted-work
  protocol; do not return to rewarding energy variance or reframe the primary
  result as topology counts after seeing outcomes. Report both separately.
- Use only the existing FIT-parent physical behavior data and the frozen36/12
  internal split. No fitting on either evaluated molecular cohort, fresh oracle
  proposal outcomes, or the722 reserved conditions. Freeze the exact ablation
  protocol before any optimization. Initial construction and offline testing need
  no additional physical queries.

Independent checks must cover symmetry, normalized action probabilities, inverse
eligibility, complete forward/reverse likelihood ratios, support and actual
importance weights. A small offline gain still needs fresh measured sampling
validation before a production comparison. Do not treat the probability-flow
identity, action logits, or an extra neural head alone as established AI novelty.

## Implementation checkpoint

The candidate is now implemented in `cfm_mol/bounded_action_geometry.py`.
`BoundedConditionalActionPolicy` reuses the old symmetric message/action encoder
and substitutes continuous atomic descriptors. It has no move-family head.
`ActionGeometryGuide` keeps inactive components frozen for each ablation;
`full_observed_densities` combines conditional action and coordinate laws and
checks the recorded uniform action-count ratio against freshly enumerated sets.
The production joint transition accepts the conditional policy explicitly and
records its complete forward/reverse action probabilities. The default old
uniform-action RNG and checkpoint behavior remain unchanged.

All 1,671 behavior attempts have their legal action sets checked. The 1,597
scored pairs have valid inverse actions; forward counts range from 4 to 36 and
reverse counts from 4 to 38. Original `ChemicalMovePolicy` state dictionaries
reproduce the previous implementation exactly. Independent NumPy action
probabilities agree on real geometry; fixed finite-state and molecular gradient,
symmetry and transition replay tests pass. These are implementation checks.

The frozen six-arm protocol is
`research/evidence/action_geometry_training_protocol_v1.json`: three variants,
two seeds, 300 fixed steps, batch eight, same objective/data and total log-ratio
bound 2. Active neural parameter counts and runtime are reported, not claimed
matched. No fresh-oracle validation outcomes enter this fitting experiment.
Training and audit scripts are `train_action_geometry.py` and
`audit_action_geometry.py` under `scripts/research/`. The audit rebuilds the split,
replays final/baseline metrics, checks both active-head gradients and independently
evaluates full densities on one scored edge per parent. Read NEXT for live jobs.
