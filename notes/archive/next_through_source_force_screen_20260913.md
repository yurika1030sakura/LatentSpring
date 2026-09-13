# Next research actions — September 12, 2026

Read `research/SOURCE_FORCE_SCREEN_STATE_20260912.json`. Use the laboratory
checkout; never write to home. The full ICLR goal remains active and unachieved.

1. Source-force screening and whole-prefix accounting are COMPLETE and audited.
   Summary: `runs/source_force_screen_summary_v1/results.json`. Both neural seeds
   lose to the fixed physical screen; their difference intervals are below zero.
   Only seed0 improves over no screen and its FIT-matched thinning control.
   Neither seed establishes benefit over linear learning. Do not scale the weights.
2. Read `notes/gate_distillation_candidate_v1.md`. The FIT-only diagnostic shows
   neural seed0 retaining98.95% of recorded expected acceptance but costing7.26
   times an inadmissible oracle teacher, versus75.67% /5.71 for seed1. These do
   not change the primary result or establish mixing. Actual candidate energy
   defines the oracle labels and is unavailable to a cheap gate at inference.
3. `cfm_mol/gate_teacher.py` and two tests implement ordinary-MH-preserving bounded
   oracle labels and a simple log-underprediction/acceptance-retention relation.
   It is an elementary identity, not a new theorem. Freeze a matched total-step
   comparison before optimization: direct utility800 versus gate-target500 plus
   utility300, both linear/neural and two seeds, with the same data, bounds and
   controls. Exact objective/sampling choices are not frozen yet. No distillation
   model has been fitted. Do not just extend the old300-step runs.
4. Both phases may use only the36 FIT parents. Keep the12 internal-selection
   parents, every failure, the original force/energy provenance and whole-prefix
   cost accounting. Source force may enter the forward gate; candidate force and
   true energy enter labels or post-query reverse correction only. Retention is
   a separate diagnostic, never a replacement success metric after outcomes.
   Initial work needs no new physical queries.
5. The existing `utility_onpolicy_v1` PHYSICAL proposals remain available for a
   separately frozen internal check of candidate gates on1,152 recorded draws.
   No fitting on these outcomes. Such a check estimates expectations, not actual
   saved calls or screened chains. Preserve its internal/reused-data designation.
6. Real chains, complete wall-time/data/query costs, strong physical and
   representative learned-generator baselines, realistic reuse and independent
   final evaluation remain necessary. Keep the old six-composition and48-parent
   molecular evaluations and all722 reserved outcomes out of fitting. Do not
   mistake ordinary screening, cached forces or a new training loss for novelty.

Training46199581 and audit46199747 are terminal; re-query Slurm before action.
Current results are fixed-source whole-prefix expectations, not changed chains.
The previous conditional-joint93% physical-screen rate gain and current13.57%
whole-prefix gain use different denominators and are not contradictory. Read
STATUS for the current PDF. Prior NEXT:
`notes/archive/next_through_delayed_screen_20260912.md`.

Verified development PDF: `runs/verification/source_force_screen_completed_20260912/main.pdf`
(9 main pages,23 total). Build evidence:
`research/evidence/source_force_screen_completed_build_20260912.json`. Scientific
submission readiness remains false.
