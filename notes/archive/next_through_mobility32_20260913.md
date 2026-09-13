# Next research actions — September 13, 2026 UTC

Read `research/MOBILITY_RELAXATION_RESULT_20260913.json` and
`notes/mobility_relaxation_decision_v1.md`. Work in the laboratory checkout;
never write to home. The full ICLR objective remains active and unachieved.

1. The paired mobility diagnostic and all four independent audits are COMPLETE:
   32 FIT parents, 128 arms, 5,564 raw physical calls. Producer46212367 and
   audit46212970 are terminal. All 2,654 queried optimization trials replay.
   Do not rerun the producer or interpret completion as physical convergence.
2. Collective-minus-root paired gap change is -0.07727 eV, descriptive parent
   interval [-0.22250, 0.08523]. Destination improvement is accompanied by source
   improvement. Only 5/64 collective versus 59/64 root-only arms converge;
   49 collective arms exhaust the cap, ten stop at the minimum step. This is
   inconclusive about intrinsic chemical gaps and exchange-specific geometry.
3. Resolve the diagnostic's finite optimization and constraint stops before
   selecting a learned architecture. Saved minimum-step events show geometric-
   domain stops in nine arms and frozen-graph stops in two; inspect actual
   active constraints before changing an optimizer. No new oracle is needed
   for that inspection. Preserve the target and graph semantics.
4. If extending optimization, first freeze a bounded continuation of ALL
   budget-exhausted arms, retaining converged and boundary-stopped arms in the
   denominator. Replay cached prefixes; preserve histories, query accounting,
   charge/spin, both mobility choices and both endpoints. Do not relaunch from
   scratch, silently increase budgets or call unconverged endpoints minima.
   No continuation protocol or job is frozen/submitted yet.
5. Optimization is teacher feasibility, not a sampler. Existing whole-COM
   escorted paths already failed their useful acceptance/cost test. Any new
   transport must have its actual density/Jacobian or a fully reversible
   auxiliary path. Do not infer novelty from collective motion or another loss.
6. Stop scaling/tuning the tested screening-only recipes. The completed larger
   48-parent, six-composition check establishes no gain over no screening.
   Scalar, geometry, action, stronger-control and transfer negatives remain.
7. Keep all evaluated cohorts and 722 reserved conditions out of fitting. The
   broader goal requires reproducible real molecular gains, complete timing/
   data/query costs, strong physical and representative learned-generator
   baselines, realistic reuse and independent final evaluation.

Summary: `runs/mobility_relaxation_summary_v1/results.json`.
Verified development PDF: `runs/verification/mobility_relaxation_20260913/main.pdf`
(9 main pages,25 total). Build evidence:
`research/evidence/mobility_relaxation_build_20260913.json`.
The paper is scientifically unready. Re-query Slurm before any recovery.
Prior NEXT: `notes/archive/next_through_gate_transfer_20260913.md`.
