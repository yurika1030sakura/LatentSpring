# Next research decision — September 12, 2026

Current authoritative checkpoint: `research/SHARP_CONTROLS_AND_TRAINING_STATE_20260912.json`.
Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. The ICLR goal remains active and scientifically unachieved.
Do not write to the old home checkout. Read `CLAUDE.md` for environment and target
contracts and `research/CLAIM_AND_BENCHMARK_SCOPE.md` for the user's actual scope.

The frozen normalized-site learner has **no established competitive advantage**.
The original 72-parent, concentration-10 comparison and its GFN2 check remain
valid for those endpoints. With the stronger concentration-64 physical prior,
the matched-total learned-minus-physical mean is +0.0320 eV, parent 95% interval
[-0.0002, +0.0673]. The stronger controls also do not establish average benefit
on the six additional compositions. Both concentration-400 calibration-accounting
regimes are retained. Do not generalize the old GFN2 result to unscored new endpoints.

All 28 new physical control arms, full trajectory replay and independent joint
MH ratios are complete. The complete control round uses 688,128 raw calls.
Summary: `runs/site_concentration_summary_v1/results.json`; reviewed figure:
`runs/site_concentration_figure_v1/controls.pdf`. The original learner's advantage
against concentration 10 is baseline-specific; do not repeat or scale its weights.

The stiffness student's internal failure is now diagnosed: direction contributes
more than 98% of held-out-parent local-teacher KL, despite good width fitting.
This is a training-context diagnostic, not a new method result or a new physics law.
See `runs/stiffness_mode_diagnostic_v1/results.json`.

Eight disjoint proposal-training compositions have now been generated and fully
audited: support counts 45, 32, 106, 99, 0, 19, 0, 96 out of 256 each, zero validator
exceptions. Both zero-support compositions are retained. The six nonempty cases
supply 96 first-supported training parents, with 64-step physical preparation and
complete replay costing 11,910 raw calls. No evaluated-six-composition coordinates,
reference initializations or reserved outcomes enter this preparation. These are
training inputs, not equilibrium samples or proof of learned benefit.

Source audit: `runs/proposal_training_geometry_audit_v1/results.json`.
Preparation audit: `runs/proposal_training_preparation_audit_v1/results.json`.
All jobs from this round are terminal at this checkpoint. Re-query Slurm before
new work; never restart a process because an observation timeout elapsed.

Current manuscript: `paper/angular_working.tex`; reviewed-build output:
`runs/verification/sharp_controls_20260912/main.pdf`, eight main pages. The abstract
and results include the failure against stronger physical controls. Build/citation
checks pass; scientific submission readiness remains false.

## Next authorized work

Implement the bounded multi-composition conditional-probe experiment in
`notes/conditional_learning_after_sharp_controls_v1.md`. Freeze the exact protocol,
input hashes, seeds and internal parent/composition split before new queries.
The proposed new-probe ceiling is 6,144 raw calls, using audited training starts.
Both passive geometry and radius must remain fixed within each angular probe set.
Preserve all proposal-support, rank and local-approximation failures, and replay
all geometry and random streams independently before fitting.

Compare the same-data angular-force and normalized-density objectives with a
small declared physical-prior regularization ablation. A model that improves
training fit but worsens held-out direction/proper-density/work checks is not
ready for another molecular pilot. No frozen learner from this checkpoint is a
successful repair. Generic vMF, regularization, force fitting and MH remain prior art.

After internal qualification, freeze the molecular comparison at actual training
cost against the strong physical trajectories and appropriate learned baselines.
The existing physical paths reach 1,024 calls per parent and can support the
planned new cost allocation without rerunning those baselines. The evaluated six
compositions must not be used to fit weights. Reserved outcomes remain untouched
until the final method, scope and benchmark are frozen. Authors and submission
are handled by the user; do not ask permission to continue authorized research.

Earlier action lists are historical: `notes/archive/next_through_transfer_checkpoint_20260912.md`.
