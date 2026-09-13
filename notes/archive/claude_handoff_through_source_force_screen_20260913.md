# Claude continuation — September 12, 2026

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `CLAUDE.md`, `research/NEXT.md`, `research/STATUS.md` and
`research/SOURCE_FORCE_SCREEN_STATE_20260912.json`. Keep environments separate;
never write to the home checkout.

The source-force screen uses general forward/reverse gate probabilities and
candidate force only after querying. Its paired-geometry neural encoder,
linear control, fixed physical/work controls and FIT-matched thinning are
implemented. All four300-step runs and independent audits are complete.
Whole-prefix neural point gains are6.06% /2.19% versus no screen, but fixed
physical gives13.57%. Both neural-minus-physical intervals are below zero;
neural-minus-linear intervals span zero. Only seed0 beats matched thinning.
No repeatable useful neural advantage is established. Do not scale the weights.

Summary: `runs/source_force_screen_summary_v1/results.json`. All model/control
metrics and6,388 supported-pair cases replay independently; max error2.09e-14.
Six trained-head FD checks have max error8.51e-10. Training46199581 and audit
46199747 are terminal. No new physical queries or actual saved calls occurred.
The prefix proxy includes initial/nonjoint cost/work but holds source trajectories
fixed. It is not a changed chain or wall-time result. Previous negatives remain.

Next read `notes/gate_distillation_candidate_v1.md`. FIT-only gate-target
analysis suggests testing dense label pretraining before utility refinement,
with a matched total-step direct-utility control. The oracle labels and simple
retention bound are implemented/tested; no distillation model or protocol is
frozen/fitted yet. Do not claim the oracle teacher is deployable: its labels use
the candidate's energy, which a cheap forward gate cannot see. Only36 FIT
parents may enter optimization; retain12 internal-selection parents and failures.

`runs/screen_force_pairs_v1` contains all1,671 pairs with audited force provenance;
`runs/screen_prefix_accounting_v1` contains the96 original trajectory constants.
Do not fit on fresh follow-up outcomes, either prior molecular evaluation cohort
or722 reserved conditions. Data construction remains18,110 raw calls. Actual
chain gains, complete costs and suitable learned/physical controls are required.
The ICLR goal is active and scientifically unachieved. Authors/submission remain
with the user. Prior handoff:
`notes/archive/claude_handoff_through_delayed_screen_20260912.md`.

Verified development PDF: `runs/verification/source_force_screen_completed_20260912/main.pdf`
(9 main pages,23 total). Build evidence:
`research/evidence/source_force_screen_completed_build_20260912.json`. Scientific
submission readiness remains false.
