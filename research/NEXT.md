# Next — a main-generator redesign, not more routing variants

CURRENT: read `research/WORK_CHAIN_STATE_20260913.json`, `research/NEXT.md`,
`notes/work_chain_pilot_v1.md` and `notes/ai_novelty_boundary_20260913.md`.

The complete-chain study and supplementary exact controls are DONE:324 trajectories,
64 raw calls each,20,736 new calls in total. Every trajectory reaches its cap;
10,044 scored MH ratios,4,826 catalogue normalizers and all state histories replay.
All project jobs from this stage are terminal. No training/evaluation job is hidden
in the background; re-query Slurm before recovery.

Cooperative routing is slower and visits fewer connectivities than strong single
edits. The single-edit learner visits2.1667 connectivities versus1.9444 uniform,
1.7500 force-informed and1.8889 root-noise, but the exact zero-learning confinement/
volume ablation reaches2.0833. Learned-minus-zero is+0.0833 with descriptive interval
[-0.0556,0.2222]; potential difference is-0.01912 eV with interval[-0.06164,0.01961]
and opposite energy signs across replicas. No material repeatable learned increment
over this exact ablation is established. Preserve the positive weaker-control and
one-step results, but do not call them ICLR-level AI novelty or overall superiority.

Stop scaling the cooperative/scorer recipe. The next bounded design task concerns
the generator's own learned distribution; see `notes/generator_redesign_brief_v1.md`.
That replacement is a hypothesis only, not implemented or validated. Ordinary
learned energy-guided MCMC, residual learning, composed paths and score-blindness
repairs have close prior art. The full ICLR goal remains active and unachieved.
Keep the evaluated12/18-parent cohorts outside fitting, all722 reserved outcomes
unqueried, optimizer recovery paused and the two software environments separate.

Concrete next task:
1. Read `notes/generator_redesign_brief_v1.md` and inspect the current BGFM target,
   conditioning and energy-loss definitions. Determine whether a valid nonlocal
   density/energy calibration mechanism can contribute something distinct from
   the existing grouped energy variance and the cited prior work. Do not infer a
   mode-weight pathology from the present short-chain results.
2. Use the existing `clamped_density.py` sampler/density pair as a possible
   controlled interface, with its actual semantics and prior numerical audits.
   Its q0.95 is not the production CTMC endpoint plus noise. Density and sampling
   must refer to one explicitly defined law; retain common conditioning across
   transported samples. No new broad density audit or oracle labels are needed
   merely to read and specify this candidate.
3. Write one concrete learned-object/architecture/objective specification and
   its minimal known-probability test. Explain the distinction from Boltzmann
   generators, SLMC, existing score/ratio methods and DiffCLF where relevant.
   A renamed pairwise variance loss is not sufficient. This is an open design
   decision, not a validated new method or a promise of ICLR acceptance.
4. If a distinct hypothesis survives, implement one bounded generator experiment
   with physical/zero-learning controls from the outset. Evaluate actual samples,
   valid retention, physical distribution claims only with a qualified reference,
   and both training/inference costs. No all-case perfection, independent new
   physics law or optimizer cleanup is required before a useful small test.

What is closed:
- The six-arm cooperative chain experiment and all three matched single-edit
  controls are complete. Do not restart them or increase budgets to seek a win.
- The known confinement term explains much of the single learner's exploration
  advantage over weaker controls. Its remaining increment is uncertain. Keep
  single_linear as a useful learned baseline, with single_restraint as its exact
  zero-learning counterpart. Do not drop that counterpart from later comparisons.
- Force-informed reverse probabilities correctly use the selected candidate's
  force AFTER its energy query. A full old-condition replay verifies that this
  adapter extension preserves earlier normalized-kernel results.
- Query-budget endpoints, connectivity visits and typed distance movement do not
  certify equilibrium sampling, thermodynamic basins or exact energy distributions.

Current artifacts:
- `research/evidence/work_chain_matched_controls_v2.json`: all9 methods with
  checked common sources, target, background, seeds and budgets. Version1 remains.
- `research/evidence/work_chain_final_table_v1.csv`: concise endpoint/runtime table.
- `research/evidence/work_chain_summary_v1.json`:216 primary trajectories.
- `research/evidence/work_chain_single_controls_summary_v1.json`:72 uniform/force trajectories.
- `research/evidence/work_chain_single_restraint_summary_v1.json`:36 zero-learning trajectories.
- `research/evidence/work_chain_move_attribution_v1.json`: accepted energy changes
  telescope by move family. Descriptive attribution, not a causal intervention.
- `research/evidence/work_chain_scheduler_hardware_v1.json`: actual Slurm records
  and A100 MIG3g.20GB node class; wall times are observed under shared-node load.
- `cfm_mol/work_chain_kernel.py`, `scripts/research/evaluate_work_chains.py`,
  `summarize_work_chains.py`, `compare_work_chain_controls.py`: reusable code.
- `runs/work_chains_v1`, `runs/work_chains_single_controls_v1`,
  `runs/work_chains_single_restraint_v1` and their matching audit directories:
  immutable complete traces, every checkpoint and raw response.

Jobs46303471/46303884,46307645/46307852 and46310519/46310594 all completed0:0.
Latest targeted tests:9 pass, plus the old mixed-chain regression and an actual
old-condition replay. No authorship/submission actions are taken for the user.
The old manuscript remains a development draft, not an ICLR-ready paper.
