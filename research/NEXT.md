# Next research actions — September 12, 2026

Read `research/ACTION_GEOMETRY_STATE_20260912.json`. Use the laboratory checkout
`/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`; never write to home. The ICLR
objective remains active and scientifically unachieved.

1. The conditional action/geometry model and production action-probability
   correction are implemented. The old policy API reproduces old weights exactly.
   All 1,671 behavior action sets and 1,597 inverse counts pass checks. Tests cover
   finite-state utility, actual-map gradients, symmetries, forward/reverse action
   probabilities and full transition replay. See the implementation evidence.
2. Finish the frozen six-arm offline comparison: training 46192301, dependent
   independent audit 46192339. Both action-only arms are complete; geometry and
   joint arms remain to be finished. Re-query Slurm and parse result JSON before
   action. Do not restart a live task or treat a temporarily empty curve as failure.
   The first action-only internal rates are below physical initialization; these
   are unaudited interim results, not a basis for altering the frozen experiment.
3. Verify all final models, split reconstruction, every final/baseline edge,
   active-head finite differences and 96 independent full densities per model.
   Use `scripts/research/summarize_action_geometry.py` to report all three variants
   against physics and joint against each component, for both seeds. The exact
   command is in the machine checkpoint. Preserve negative results and intervals.
4. Training uses only the frozen 36 FIT parents, with 12 internal-selection
   parents; all failed attempts remain. Total likelihood-ratio bound exp(2),
   data, optimizer, steps and minibatch draws are matched. Active parameter counts
   and runtime are reported separately. No new physical queries are needed.
   Data construction costs remain 18,110 raw calls.
5. An offline signal alone cannot qualify scale-up. Any next physical experiment
   must be frozen before outcomes and measure actual sampling utility against
   appropriate physical and learned controls. Do not fit on fresh follow-up
   outcomes, the prior six-composition evaluation coordinates, the 48-parent
   scalar-chain cohort, or the 722 reserved outcomes. Representative learned
   generators, realistic reuse costs and independent final evaluation remain
   required on the declared scope.

The previous geometric utility model had about 2.7% offline gain but no established
fresh-proposal benefit. The scalar-chain and stronger-control negatives remain.
Read `research/ACCEPTED_UTILITY_STATE_20260912.json` for those complete results.
The current development PDF is unchanged:
`runs/verification/accepted_utility_20260912/main.pdf` (8 main pages, 20 total).
It is not scientifically submission ready.
Prior NEXT: `notes/archive/next_through_geometric_utility_20260912.md`.
