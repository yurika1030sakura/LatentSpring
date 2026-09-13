# Next — validate cooperative interaction learning in the full sampler

CURRENT: read `research/EDIT_INTERACTION_STATE_20260913.json`, `research/NEXT.md`
and `notes/cooperative_interaction_work_v1.md`.

The cooperative-interaction prototype is implemented and fully tested. Four-state
FIT data reveal real electronic coupling (parent-balanced mean absolute0.16419 eV).
All12 interaction models train/replay; the internal diagnostic has18 fit/9 held
parents, with9 original parents having no pair kept separately. Full models fit27
parents. The3D radial-contrast representation learns coupling; environmental
conditioning has no demonstrated advantage over simpler models.

A fixed-root-block pilot is complete and retained: many matchings are equivalent
under same-element atom relabelling, and utility differences are negligible.
The retained-panel kernel now chooses across different root blocks, uses the
SAME panel in the reverse normalizer, and treats coupling as a symmetric edge
preference, NOT a directional total-work correction. All physical states and
4,400 MH ratios replay. This gives small2--4% point improvements within the
cooperative family, with uncertainty and no contextual-network superiority.
The strongest single-edit learned method still has higher mean one-step utility.
There is no overall generator/chain advantage or ICLR-readiness claim.

All current jobs are COMPLETE0:0:46296885/46297013 labels/audit,46297734 training,
46299227/46299395 fixed-block evaluation/audit,46299770/46299887 panel replay/audit.
This stage used1,770 new raw calls (934 labels+836 endpoint evaluation); panel
re-evaluation used zero. Keep722 reserved outcomes unqueried and evaluated parents
out of fitting. Optimizer recovery and old single-edit neural scale-up stay paused.
The full ICLR goal is active and scientifically unachieved.

Completed method and evidence:
- `cfm_mol/chemical_edit_interaction.py`: commuting double edits, mixed electronic
  work, known COM-restraint cross term and intrinsic map volume.
- `cfm_mol/interaction_work_model.py`: four-corner radial contrasts with typed
  linear, context-blind neural and environmental coefficients. Reversing one
  edit negates interaction; reversing both preserves it. Do not add interaction
  directly to directional work; use the symmetric affinity implementation.
- `cfm_mol/cooperative_edit_policy.py`: type-admissible root blocks, complete valid
  matchings and retained random panels. The panel draw law is invariant under
  the prescribed edit, so it cancels. Forward/reverse catalogue probabilities
  must still both be recomputed. Empty panels are self-loops.
- `scripts/research/evaluate_cooperative_edits.py`: physical evaluation and
  query-exact panel re-evaluation. `summarize_cooperative_edits.py` consumes only
  complete audited outputs. The two scopes are different kernels; their physical
  responses match, while their selection outcomes need not.

Latest quantitative limits:
- Original36 FIT parents:27 have pairs,9 have none.450 selected pairs,440 valid,
  10 failures.24/440 reverse the sign of an oracle-additive work estimate; only
  one has both singles uphill and the joint edit downhill. This is mechanism
  data, not evidence that all proposals bypass a barrier.
- Internal zero-interaction MAE0.1995 eV; typed radial0.1667/0.1703; blind neural
  0.1557/0.1555; environment0.1549/0.1579. On15 supported evaluation parents,
  zero0.1922, typed radial0.1415/0.1382, blind0.1673/0.1558, environment0.1736/0.1595.
  No evaluated endpoint label was fitted. Keep the simple radial model as a
  strong control instead of selecting only a favorable neural comparison.
- Mean panel one-step utility: uniform0.051304, additive linear0.086910,
  restraint-only0.086871, radial interaction0.089929, blind interaction0.089231,
  environment0.089108 eV. The blind-minus-restraint descriptive interval is
  [0.000287,0.004940]; radial/environment intervals cross zero. These are small
  repeatedly evaluated development signals, not a model-selection certificate.
- Prior strong single-edit linear utility was0.121892 eV on the same18 starts.
  The current cooperative operator does not win that energy-descent comparison.
  The six-composition denominator retains three parents with no double edit.

Next bounded experiment (not yet implemented or submitted):
1. Integrate the existing cooperative panel kernel into a short complete-chain
   comparison with the same background moves, query caps and starting states.
   Include strong single-edit linear selection and cheap physical root-noise
   moves, cooperative linear selection, typed radial coupling and blind coupling.
   Environmental-network scale-up is not the priority.
2. State the finite-temperature exploration/basin-transition hypothesis before
   seeing new outcomes, and also retain energy descent and query/wall-time costs.
   A cooperative shortcut may affect exploration differently from early energy
   descent, but that is presently a hypothesis, not a reason to discard the
   negative single-edit comparison or select favorable molecules.
3. Reuse audited existing source/late-chain checkpoints where appropriate;
   no all-case optimizer cleanup or new static map is required. Freeze the
   source role/selection and protocol before queries. Fit no evaluation parent
   and query none of the722 reserved outcomes. Choose a small fixed budget,
   two seeds and matched controls; no acceptance/ICLR promise.
4. If no material complete-chain advantage emerges, stop scaling this recipe.
   Generic mixed differences, composed paths, auxiliary MH and learned radial
   coefficients are prior art building blocks. The useful learned mechanism and
   comparisons must support the paper's actual novelty claim.

Key artifacts:
- State: `research/EDIT_INTERACTION_STATE_20260913.json`.
- Data: `runs/chemical_edit_interaction_plan_v1`, `runs/chemical_edit_interaction_v1`,
  `runs/chemical_edit_interaction_audit_v1`.
- Models: `runs/interaction_work_training_v1/s{0,1}/full_fit/{linear,blind,environment}/model.pt`.
- Fixed-block outputs: `runs/cooperative_edit_evaluation{,_audit}_v1`.
- Panel outputs: `runs/cooperative_panel_evaluation{,_audit}_v1`.
- Evidence summaries: `chemical_edit_interaction_summary_v1.json`,
  `interaction_work_training_summary_v1.json`, `interaction_work_transfer_error_v1.json`,
  `cooperative_edit_evaluation_summary_v1.json`, `cooperative_panel_evaluation_summary_v1.json`
  under `research/evidence/`. All former results and source snapshots remain.

No project jobs are running at this checkpoint; re-query Slurm before recovery.
The old manuscript is a development draft and needs later rewriting around an
actually supported method. Do not spend the next step polishing its layout.
