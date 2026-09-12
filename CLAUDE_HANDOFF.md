# Claude continuation — September 12, 2026

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `CLAUDE.md`, `research/NEXT.md`, `research/STATUS.md` and
`research/ACTION_GEOMETRY_STATE_20260912.json`. Keep environments separate and
never write to the home checkout.

The six-arm conditional action/geometry comparison is COMPLETE and independently
audited. Action-only internal rates fall 1.94% / 1.74%; joint rates fall
0.21% / 0.18%. Both variants' intervals versus physics span zero. Geometry-only
reproduces the preceding +2.7% offline proxy; its actual fresh-proposal benefit
remains unestablished. Joint has no demonstrated gain over geometry alone.
Do not scale these weights. All source failures and previous negatives remain.

Every final/baseline metric replays, with 576 independent full density checks
(max error 4.98e-14) and eight trained-head finite differences (max error
3.17e-10). Production tests cover simultaneous learned action and geometry,
NumPy action probabilities, arc quadrature and trajectory replay. Geometry-only
parameters reproduce the prior models within 7.8e-16. Jobs 46192301 and 46192339
are terminal; re-query Slurm before action. Summary:
`runs/action_geometry_summary_v1/results.json`.

A FIT-only cost census finds that expected rejected joint calls account for
24.35% of the recorded complete-trajectory call budget. This is not an achievable
saving or a wall-time prediction. The oracle-informed screening diagnostic is
explicitly inadmissible as a cheap algorithm. Next read
`notes/delayed_acceptance_prior_art_20260912.md` and NEXT before implementing any
screen. Delayed acceptance, including neural and structural-move variants, has
prior art. It cannot be renamed as our novelty.

A possible bounded comparison must preserve exact reverse pairing, original
charge/spin/support and the paired oracle, start at the physical kernel, and
include cheap physical and simple learned screens. Only the 36 FIT parents may
enter fitting; retain the 12 internal-selection parents and all failed attempts.
No fitting on fresh follow-up outcomes, either prior evaluated molecular cohort
or the 722 reserved conditions. No new physical queries were used in this round.
Data construction still costs 18,110 raw calls.

The development manuscript now includes the six-arm negative result in its
main-text interpretation and appendix. The ICLR goal remains active and the
paper scientifically unready. Authors and actual submission remain with the user.
Prior handoff: `notes/archive/claude_handoff_through_action_geometry_running_20260912.md`.

Verified development PDF: `runs/verification/action_geometry_completed_20260912/main.pdf`
(9 main pages, 21 total). Build record:
`research/evidence/action_geometry_completed_build_20260912.json`. Scientific
submission readiness remains false.
