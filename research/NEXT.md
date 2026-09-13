# Next research actions — September 13, 2026 UTC

Read `research/GATE_DISTILLATION_RESULT_20260913.json`. Work in the laboratory
checkout; never write to home. The full ICLR objective remains active and unachieved.

1. Matched800-step direct/distillation training and all8 audits are COMPLETE.
   Pretraining loses to direct utility on the original12 selection parents.
   The larger48-parent, six-composition evaluation and all6 audits are also
   COMPLETE. No tested screen establishes a benefit over no screening there.
   Do not scale or keep tuning these screening-only recipes. Small favorable
   comparisons against another screen do not replace the no-screen control.
2. Read `notes/post_screening_reconstruction_20260913.md`. Inspect existing
   `escorted_exchange.py`, `chemical_path_guide.py`, their v1/v2/v3 experiments,
   and `notes/nonequilibrium_chemical_candidate.md` before proposing collective
   relaxation. Whole-COM paths already exist; v1/v2 accepted only4/64 paths per
   replica even after endpoint support improved. Preserve those negatives.
3. Make the next physics question discriminating before training another network:
   distinguish geometric restriction from intrinsic chemical energy gaps and
   proposal/path-probability costs. A bounded FIT-only diagnostic can compare
   root-only and collective relaxation of BOTH the source and exchanged endpoint,
   with matched oracle budgets and fixed perceived connectivity. Large anchor
   forces alone are not proof, and lowering a destination alone confounds generic
   relaxation of the source. Freeze exact selection, optimizer, query limits,
   convergence/support checks and failure denominators before new calculations.
   No such new protocol or job is frozen/submitted yet.
4. Treat optimization as teacher feasibility, not a sampler. Any later collective
   learned proposal needs its actual density/Jacobian or a complete reversible
   auxiliary path; swap-plus-uncontrolled-relaxation with endpoint-only MH is
   incorrect. Only proceed to a new learned framework if the diagnostic identifies
   an opportunity. Do not assume another loss, neural head or collective move is
   intrinsically novel or useful.
5. The larger screening cohort is EVALUATION ONLY:48 parents excluded from gate
   fitting and original12-parent selection, including two unseen composition
   identities. It was used earlier for scalar evaluation and is not a final test.
   Do not fit on it, fresh-proposal outcomes, any other prior evaluated cohort or
   the722 reserved conditions. Preserve all source failures and electronic states.
6. The final goal still requires reproducible real molecular gains, complete
   timing/data/query costs, strong physical and representative learned-generator
   baselines, realistic reuse and independent final evaluation. Current expected
   substitutions at recorded prefixes are not changed chains or actual savings.

Jobs46203103,46203225,46206100 and46206282 are terminal. Re-query Slurm before
recovery; do not restart completed work. Summaries:
`runs/gate_distillation_summary_v1/results.json` and
`runs/screen_transfer_summary_v1/results.json`. Read STATUS for the verified PDF.
Prior NEXT: `notes/archive/next_through_source_force_screen_20260913.md`.

Verified development PDF: `runs/verification/gate_distillation_transfer_20260913/main.pdf`
(9 main pages,24 total). Build evidence:
`research/evidence/gate_distillation_transfer_build_20260913.json`. Scientific
submission readiness remains false.
