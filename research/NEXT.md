# Current AI-method work — a useful simple baseline, neural value unresolved

Read `research/CHEMICAL_WORK_POLICY_STATE_20260913.json` and
`research/evidence/chemical_work_policy_evaluation_summary_v1.json`.
User priority remains differentiated AI plus fast molecular evidence. No all-case
perfection or optimizer recovery prerequisite. The ICLR goal remains active.

1. The paired masked-context model is IMPLEMENTED in `cfm_mol/chemical_work.py`.
   Both moved atoms are removed from passive encoding. A shared two-root decoder
   evaluates both placements; their difference reverses sign. The graph-only
   ablation removes geometric inputs except the known restraint. The additive
   typed-bond model is a strong learned control. No one-pass cache across active
   masks or global cycle consistency is claimed.
2. Two seeds of all three variants each have24/12-parent diagnostic fits and
   separate36-parent full fits,400 fixed steps each. All12 models and their
   identical parent streams replay. No new labels:447 existing valid FIT edits.
   Internal3D MAE0.683/0.704 eV loses to linear0.566/0.565 eV. Full models must
   not be evaluated as held-out on the internal12, since those enter full fits.
3. `cfm_mol/chemical_work_policy.py` enumerates the actual valid catalogue and
   recomputes normalized reverse probabilities. Forward inference has no true
   candidate E/F. The force control may use the physically scored selected
   endpoint force for its reverse probability. Uniform mixture=.1; actual map
   volume remains explicit. Eight focused tests pass.
4. The18-parent molecular evaluation is COMPLETE,498 new raw calls,231 valid
   and20 invalid actions. All catalogue endpoints were scored FOR EVALUATION;
   no deployed oracle-assisted forward ranking is implied. Exact one-step
   expected utility is0.070621 uniform,0.062508 force,0.121892 linear (two-seed
   mean),0.082167 graph and0.0883913D. Linear beats uniform/force in this limited
   comparison;3D has no established advantage over uniform, graph or linear.
   Full replay and1,848 independent physical MH ratios pass. Keep all cases.
5. Retain linear work selection as the strong learned baseline. The next compact
   neural hypothesis should include that additive predictor explicitly and
   learn the non-additive3D residual, using the same split and a frozen bounded
   protocol. This is an architectural repair, not by itself a novelty claim.
   Require a useful representation contribution beyond linear/graph controls
   before expanding experiments or rewriting the paper around it. Do not tune
   on evaluation labels. No further labels are needed to start that prototype.
6. A later full-chain experiment must compare against valid-catalogue uniform,
   cached force, linear work and the effective bare/root-noise controls, count
   data/preparation and inference costs, and retain appropriate generator
   comparisons for the paper. One-step work selection is not evidence of target
   energy-distribution sampling or a complete generation improvement.
7. Original12 and new18 evaluation parents are development/evaluation-only;
   never fit them.722 reserved outcomes remain unqueried. The user's priority
   does not require new physics laws, universal perfection or optimizer cleanup.

Artifacts:
- Training: `runs/chemical_work_training_v1`, protocol and summary named
  `research/evidence/chemical_work_training_{protocol,summary}_v1.json`.
- Molecular evaluation: `runs/chemical_work_policy_evaluation_v1`; full replay:
  `runs/chemical_work_policy_evaluation_audit_v1`; protocol/summary in evidence.
- Model files: `runs/chemical_work_training_v1/s{0,1}/full_fit/{linear,graph,geometry}/model.pt`.
- Reproducible audit scripts: `scripts/research/audit_chemical_work_training.py`
  and `scripts/research/summarize_chemical_work_policy.py`; use new output paths.

All current jobs are terminal:46258095 training,46258563 evaluation,46258596 audit.
Re-query Slurm before recovery. No automatic rerun or model scaling is authorized
by a stale status paragraph. The old mobility result and manuscript stay archived;
no PDF-formatting work is needed before useful AI evidence.
