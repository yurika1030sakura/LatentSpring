# Next — component-mass calibration checkpoint

Read `research/LATENT_MASS_STATE_20260913.json` and
`notes/latent_mass_calibration_v1.md` first. The ICLR goal remains unachieved.
User priority is useful AI novelty and quick evidence, not all-case perfection.

The new prototype learns Gaussian-preserving couplings between normalized
reference generators to reduce component-mass estimation error. It is implemented
and tested on exact-probability COM targets. It keeps each conditional reference
shape fixed. Generic ratio coupling and Gaussian-preserving rearrangements have
direct prior art; do not claim those principles as new.

Completed evidence:
- Offline run46315997, audit46317683 and binned control46317713 are complete.
  At64 pairs, nonlinear-case mass RMSE averages0.03884 independent,0.02285 neural
  and0.02281 binned. Simple nonlinear coupling matches neural with fixed fitting
  data. The constant-angle case gives no substantive neural increment.
- The online experiment46320232 is complete:384 histories across two constructed
  cases and six methods,512 target calls per history including all learning.
  Every query is retained. Audit46320415 is complete. All17 array tasks from this
  stage completed0:0. No stage job remains active; re-query Slurm before recovery.
- No molecular oracle calls were added. These COM shells are not chemically
  validated molecules. Correct component masses alone can worsen joint reverse
  KL when conditional shapes remain misspecified; that result is retained.

The final online nonlinear-case mass RMSE is0.016519 independent,0.016149 shared
latent,0.017523 constant,0.017964 Gaussian,0.012027 neural and0.014159 binned.
Neural point error is27.2% below independent and15.1% below binned, but both paired
MSE intervals span zero. Constant/Gaussian maps have lower point error than neural
on the constant case. The binned control matches neural offline. Do not claim a
robust neural win or molecule-generation advance from these results.

Final artifacts:
- `research/evidence/online_latent_mass_summary_v1.json`: all methods, budgets,
  paired uncertainty, source hashes and full-run compute times.
- `research/evidence/online_latent_mass_table_v1.csv`: flat numerical table.
- `research/figures/latent_mass_v1/online_latent_mass_v2.pdf`: curves and paired
  uncertainty. Version1 point-only figure remains preserved.
- `runs/online_latent_mass_v1` and `runs/online_latent_mass_audit_v1`: complete
  traces and audits. All196,608 weights and prefix estimators replay;48 histories
  replay end to end, including32 adaptive and16 static histories.
- `research/evidence/latent_mass_scheduler_v1.json`: actual completion records.

Next main-method decision:
1. Retain the coupling as an estimator prototype. It cannot change marginal
   importance-weight collapse or repair conditional geometric coverage. A new
   paper story cannot be built on generic coupled importance sampling alone.
2. Choose one explicit learned-distribution contribution and a small molecular
   test that directly measures it. The normalized sampler/density interface and
   strong simple baseline must be part of that specification from the start.
   The present module does not establish that such a contribution is solved.
3. Do not increase toy budgets, change targets, add neural variants or revive
   routing sweeps merely to seek a favorable comparison. Do not polish the old
   diagnostic manuscript into a claimed new method before that decision.

A molecular continuation must first choose one normalized reference sampler with
its matching density, within one fixed composition/electronic sector. Production
FM64 plus noise does not have a qualified absolute q; clamped q0.95 is a different
law. Existing EACF checkpoints are implicit-source refiners, not established
absolute-density reference banks. Historical triatomic integrals differ in target,
spin, temperature or support and cannot silently serve as ground truth here.
These are concrete interface choices, not a request for another broad density
or all-case chemistry audit. Keep the first actual molecular test small and
include a simple nonlinear control and a learned-generator baseline.

The earlier routing branch is closed to scale-up. Its324 trajectories and20,736
calls are archived under `research/WORK_CHAIN_STATE_20260913.json` and
`research/evidence/work_chain_matched_controls_v2.json`. The learned single edit
has no established material increment beyond its exact zero-learning counterpart;
cooperative routing is slower and explores fewer connectivities. Preserve every
negative result. Do not restart scorer/committee/mobility/optimizer sweeps.

Protected constraints:722 reserved outcomes unqueried; evaluated12/18-parent
cohorts excluded from fitting; two environments separate; no home writes;
bond-free OMol25 with max_atoms200; user handles authorship/submission.
`paper/angular_working.tex` remains the old diagnostic draft, not the current
method paper or an ICLR-ready submission.
