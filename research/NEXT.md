# Next research actions — September 12, 2026

Read `research/CONDITIONAL_NONLOCAL_STATE_20260912.json` and `research/STATUS.md`.
Use `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`. Never write into the home
checkout. The ICLR goal remains active and scientifically unachieved.

1. The scalar model,438-context dataset, four800-step trainings and independent
   learning audit are COMPLETE. Do not retrain or repeat the physical pilot.
   Training: `runs/conditional_arc_train_v1/{work,work_force}_s{0,1}`.
   Audit: `runs/conditional_arc_learning_audit_v1/results.json`.
   The `arc_energy` complete joint decoder is implemented with independently
   checked scalar energy readout and forward/reverse normalization.
2. Implement the frozen `research/evidence/conditional_arc_chain_protocol_v1.json`
   complete-chain pilot on all48 internally withheld parents:16 each
   in conditions0 and7, four each in1,2,3,5. Their IDs are the union of the two
   withheld lists in `multicomposition_angular_probe_protocol_v1.json`. Keep both
   learned objectives/seeds and a physical site-arc control; no favorable-seed
   selection. These are internal development outcomes, not the reserved benchmark.
3. Include128-call learned readouts and a precisely cost-matched physical budget.
   Shared warm preparation is6,088 calls. Model-specific data overhead is14,862
   calls (5,822 FIT-parent preparation+5,536 probes+1,648 FIT endpoints+1,856
   validation endpoints). Thus each learned method/replica costs21,006 calls plus
   common costs. A physical control with39 parents at438 calls and9 at436 costs
   exactly21,006; assign the extra pair of calls by a prospectively fixed hash.
   Retain same-inference32/64/128-call readouts and measured runtime. Maximum new
   production cost for three methods and two replicas is66,588 raw calls.
   The driver needs explicit per-parent caps and scalar-model loading; do not
   silently reuse the old fixed-cap/FIT-only driver with mismatched semantics.
4. Replay every chain, all raw-call ledgers and independent joint MH ratios before
   interpretation. Predictive error improvement alone is insufficient. If learning
   adds useful sampling, test a broader frozen reuse cohort and appropriate
   learned baselines; preserve every adverse cost regime and composition.
5. Address the prior-art boundary in `notes/nonlocal_arc_prior_art_20260912.md`.
   Classical biased regrowth, self-learning MC, learned decoders and geodesic
   sampling already exist. The intended contribution is a useful learned coupling
   of molecular connectivity/geometry and feasible conditional proposals. Consider
   classical multi-trial regrowth and same-score local proposals as discriminating
   controls; generic force fitting or a cheap neural potential is not enough.

Do not fit on the previously evaluated six-composition coordinates. Keep722
reserved outcomes untouched until the method, scope and final benchmark are
frozen. Authors and actual submission remain with the user. No new permission is
required for authorized research work. Prior NEXT:
`notes/archive/next_through_geodesic_physics_20260912.md`.
