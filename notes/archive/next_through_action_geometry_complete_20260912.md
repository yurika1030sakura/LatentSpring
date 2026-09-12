# Next research actions — September 12, 2026

Read `research/ACTION_GEOMETRY_STATE_20260912.json`. Use the laboratory checkout;
never write to home. The ICLR objective remains active and scientifically unachieved.

1. The six frozen action/geometry arms and all independent audits are COMPLETE.
   Read `runs/action_geometry_summary_v1/results.json`. Neither action-only nor
   joint establishes improvement over physics. Geometry-only reproduces its
   prior offline proxy but has no established fresh-proposal advantage. Do not
   rerun, enlarge the networks, or scale the frozen weights based on these results.
2. Read `notes/delayed_acceptance_prior_art_20260912.md` and the FIT-only
   `runs/joint_query_cost_diagnostic_v1/results.json`. Scored joint proposals have
   5.35% balanced expected acceptance; expected rejected joint calls account for
   24.35% of recorded complete-trajectory calls. Preparation and learning costs
   are additional. An oracle-informed screen is only an inadmissible information
   diagnostic. Actual saved calls are zero; separate RPC/geometry timing is absent.
3. A next bounded investigation can test cheap pre-screening with the exact
   two-stage correction. First implement finite antisymmetric factors and prove
   the actual transition's reverse pairing numerically on finite and molecular
   examples. Preserve the base proposal and fixed move-family schedule. Starting
   with factor zero must recover ordinary MH. Never discard proposals using an
   uncorrected classifier or use the candidate's expensive energy before deciding
   whether to request it. This is established delayed-acceptance mathematics.
4. If the implementation/cost case warrants fitting, freeze a small comparison of
   a cheap physical surrogate, a simple fitted screen and a neural paired-state
   screen, with common bounds/data/objective. Use only the 36 FIT parents and the
   fixed 12-parent internal-selection split. Retain failures and signed work;
   do not switch to acceptance alone after outcomes. No new physical queries are
   needed initially. Existing energy models trained on selection parents cannot
   be silently reused as held-out predictors. No screening protocol is frozen yet.
5. Any useful offline signal still needs fresh measured chain utility, query and
   complete wall-time costs. Standard delayed acceptance alone does not establish
   ICLR novelty. Strong physical and representative learned-generator baselines,
   realistic reuse costs and independent final evaluation remain required on the
   declared scope. Do not fit on fresh follow-up outcomes, prior six-composition
   coordinates, the 48-parent scalar-chain cohort or any of 722 reserved outcomes.

All current BGFM jobs are terminal: training 46192301 and audit 46192339.
Re-query Slurm before action. The scalar-chain, old vector/strong-control and
geometric fresh-proposal failures remain. The new appendix is
`paper/sections/A6_action_geometry.tex`; see STATUS for the verified build.
Prior NEXT: `notes/archive/next_through_action_geometry_running_20260912.md`.

Verified development PDF: `runs/verification/action_geometry_completed_20260912/main.pdf`
(9 main pages, 21 total). Build record:
`research/evidence/action_geometry_completed_build_20260912.json`. Scientific
submission readiness remains false.
