# Next bounded learning experiment after the full-cost control

Status: design, not implemented or validated. Read the completed decision in
`notes/joint_geometry_decision_v1.md` first. Do not relaunch the frozen pilot or
claim a learned advantage from its shorter step count.

## Evidence that the next model must address

- Joint tensor proposals reach the diagnostic graph for all four parents in both
  replicas. This repairs the earlier fixed-geometry failure.
- A coordination-site physical prior also reaches all four, at cumulative costs
 6380/2994 rather than tensor11296/11076. The9660-query inherited training bill
  dominates this four-parent comparison. Reusing a model may amortize that cost,
  but a projection based on these same parents is not empirical confirmation.
- The normalized mixture has tensor force-fit errors575.2/602.7, different from
  the quadratic-score training objective's439.7/475.0.
- The17-atom geometry-only screen gives tensor44/128 and0/128 supported endpoints,
  versus site101/128 and96/128. O/P/Cl embeddings were never trained. This is not
  an independent energy/sampling test, and it argues against blind zero-shot scale-up.
- Restrained-potential Rhat1.802/1.906 and shape disagreement prevent an equilibrium
  claim from the tensor pilot. Graph first passage must remain a diagnostic.

## Concrete implementation to test

1. Retain the normalized joint coordinate interface, physical target, random root
   order, exact Cartesian radial measure and complete reverse context reconstruction.
   Preserve the strong site-prior control with the same action list and schedule.
2. Learn corrections to the physical directional prior, using a fully normalized
   finite vMF mixture. Predict equivariant component natural parameters and
   invariant mixture weights from masked passive geometry. Avoid a change between
   the directional law being fitted and the law subsequently sampled. A finite
   mixture has an explicit logsumexp density and sphere-score derivative.
3. Include a fixed positive-probability physical-kernel component, with its own
   correct MH step, in EVERY learned arm and the matching ablation. This is a
   standard defensive mixture, not a novelty claim or evidence of efficiency.
   If a marginal mixture proposal is used instead, evaluate the FULL mixture in
   both directions; the density of only the sampled component is insufficient.
4. Replace unconstrained, never-trained element embeddings with explicit atomic
   descriptors or train the relevant elements on generated TRAINING data. A
   representation change alone does not certify transfer. Keep electronic-state
   conditioning and all source validity denominators.
5. Fit the ACTUAL normalized angular score on the existing training table first,
   with no new physical queries and a prospectively bounded CPU budget. Compare
   to the exact physical-prior initialization and a vector residual ablation.
   Check zero-concentration gradients, mixture normalization, O(3)/permutation
   symmetry and sequential reverse densities before any new molecular rollout.
6. If exact-density force fitting is insufficient, evaluate the already tested
   support-aware MH utility gradient on generated training proposals only. A
   bounded new training-query protocol must be frozen before outcomes. Preserve
   zero utility and zero physical cost for unsupported candidates; do not train
   on these four development trajectories or on QC reference geometries.

## Scientific gates before more scaling

Use the same strong prior with matched total costs. Any amortized-use claim needs
fresh disjoint generated parent batches and must count the one-time preparation,
each new source attempt, warm-up, oracle calls and wall time. Do not retrofit the
existing four-parent study into an amortized benchmark.

Train/evaluate more than one composition before opening the reserved evaluation
set. The current eligibility census supplies condition4 without energy ranking;
it does not justify dropping the six other strata from reported coverage.
Radicals and metal-validator errors require their own explicit target/domain work.

For energy-distribution claims, obtain longer frozen chains with independent
initialization families and inspect restrained energy, invariant shape and slow
conformational features. Keep early-time failures and correlated-sample uncertainty.
Lowest endpoint energy and successful graph conversion are not optimization
targets for a300K equilibrium ensemble.

Stop expansion if the revised learner cannot improve the strong prior after
the relevant full or prospectively measured amortized costs. The next architecture
and generic MH/mixture identities are candidates, not established ICLR novelty.
