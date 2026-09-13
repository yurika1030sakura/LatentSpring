# Next research actions — AI framework priority

Latest update: training46215808 is COMPLETE for all six models. Edit-conditioned
and blind collective fitting errors are nearly equal; no clear conditioning
benefit is established. Root-only fitting error is lower. Final model hashes and
matched sampled-index/direction streams are checked. Next test actual molecular
proposals; neither fitting MSE nor these core tests prove a useful AI contribution.

Read `research/AI_FRAMEWORK_STATE_20260913.json` and
`notes/edit_conditioned_bridge_v1.md` first. The user's latest correction is
recorded in `research/CLAIM_AND_BENCHMARK_SCOPE.md`: prioritize the whole AI idea
and a quick, credible demonstration. Further optimization of all difficult cases
is not a prerequisite. The full ICLR goal remains active and unachieved.

1. The new edit-conditioned reversible transport is implemented. Its neural
   field conditions on both molecular graphs and moves all atoms through
   reversible phase-space shears around a chemical edit. Four essential tests
   cover the inverse, actual intrinsic volume, MH-ratio reversal, symmetries,
   gradients and the root-only ablation. Physical queries are endpoint-only
   once a sampling caller is integrated; that caller is not implemented yet.
2. Training46215808 is COMPLETE: two seeds, three variants,300 steps each.
   Re-query Slurm. Do not restart from a missing observation or duplicate an
   existing run. Models: edit_collective, blind_collective and edit_roots.
   Data:192 paired bounded observations from32 FIT parents in four compositions;
   all original parents and failures remain. No new physical labels are needed.
3. Freeze the final checkpoints and promptly integrate the actual augmented-MH
   molecular caller. Enforce expected endpoint graph and inverse-action
   eligibility, include the momentum and chemical-map volume/action ratios,
   and keep rejected/unsupported candidates. No unknown FM density is needed.
4. Run the first real corrected-proposal pilot on the existing12 INTERNAL
   evaluation parents, excluded from fitting. Use matched source states and
   proposal/noise streams; include the physical site/arc and zero/analytic bridge
   controls as well as both architectural ablations. Measure useful accepted
   structural movement and energy-aware output at actual oracle/time costs.
   Acceptance alone or training MSE is insufficient. This is internal evidence,
   not the independent final benchmark.
5. If signal supports edit conditioning and full-coordinate transport, run short
   matched-budget chains and a small fresh-composition check. Then expand the
   evidence around the observed claim. Do not wait for every optimizer/chemical
   case to converge. If there is no signal, reconsider the framework rather than
   adding another screen head or returning automatically to the19-arm repair.
6. No further mobility/constraint-recovery jobs are planned. Completed results
   remain in `research/MOBILITY_CONTINUATION_RESULT_20260913.json`; the old19-arm
   recovery plan is superseded by this user priority. Preserve all negative data.
7. Generic neural HMC, conditional flow proposals and learned nonequilibrium
   protocols are prior art. The candidate contribution is a useful edit-conditioned
   collective transport mechanism; its novelty and value are still hypotheses.
   Keep flow matching as initializer, OMol25 primary, max_atoms200, bond-free
   supervision, original electronic states and all evaluation splits. Never fit
   on the722 reserved outcomes. Never write home or merge the environments.

Training results: `runs/edit_bridge_training_v1/<variant>_s<replica>/results.json`.
Prior NEXT: `notes/archive/next_before_ai_priority_20260913.md`.
The prior26-page PDF records completed diagnostics, not a tested new bridge.
Do not spend the next turn polishing that PDF before the AI pilot.
