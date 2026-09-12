# Next research actions — September 12, 2026

Read `research/GEODESIC_RECONSTRUCTION_STATE_20260912.json` and `research/STATUS.md`.
Use `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`; never write into the home
checkout. The ICLR goal remains active and scientifically unachieved.

1. The physical full-chain pilot46176878 and audit46176906 are COMPLETE.
   `runs/joint_arc_budget_summary_v1/results.json` includes all24 arms,48 parents,
   two seeds and36,864 calls. Site arcs improve the fixed-training-panel128-call
   potential readout by0.07615 eV against legacy site64, with22.4% extra sampling
   time; uniform arcs do not establish an average full-chain gain. Preserve all
   three query readouts and composition estimates. Do not rerun these paths.
2. Implement a masked conditional score of nonlocal energy/work along feasible
   arcs. First use existing audited FIT-context force/work and valid arc endpoint
   labels; local fit accuracy alone does not qualify long-arc probabilities.
   Freeze the feature/score and internal validation design before fitting. This
   new score is not implemented or trained yet. Keep the physical site-arc score
   as the mandatory baseline and preserve exact density evaluation on reversal.
3. Check withheld internal parents/compositions for conditional work and proposal
   quality before a bounded learned complete-chain experiment. Any extra physical
   labels need their own frozen budget/protocol. The unrestricted local teacher
   fails support, and uniform arcs' root-only signal did not carry over to full
   chains; do not use teacher imitation or acceptance alone as a release gate.
4. After an internally qualified model, freeze a matched-cost test against strong
   physical AND appropriate learned baselines. Re-account all development/training
   costs and initializer failures. Old legacy physical paths cannot be described
   as if produced by the new arc kernel. Keep the evaluated six compositions out
   of training and722 reserved outcomes untouched until final method/scope freeze.

The geometric kernel and exact normalization are foundations, not established AI
novelty. A scoped useful learning contribution is the goal; universal perfection
and a new physics law are not requirements. See `research/CLAIM_AND_BENCHMARK_SCOPE.md`.

Theory: `notes/support_constrained_geodesic_proposal.md`.
New protocol: `research/evidence/joint_arc_budget_protocol_v1.json`.
Prior execution list: `notes/archive/next_through_sharp_training_20260912.md`.
