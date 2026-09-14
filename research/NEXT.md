# Next — tree prior reduces fragmentation; learned increment remains open

Read `research/TREE_PRIOR_STATE_20260914.json` and
`notes/tree_prior_results_20260914.md`. The ICLR goal is active and unachieved.
The user prioritizes a useful AI framework and a fast molecular demonstration,
not every case perfected. Existing OMol25 is enough for the initial tests.

The bounded studies are complete. All6,144 initial source draws and final
structural readouts were replayed; all5,120 stored energy/force rows and their
source conditions/support masks were checked. Ten Slurm tasks completed0:0 and
no tree-study job was active at the recorded scheduler check. Verify live state
before recovery. Earlier rejected/cancelled submissions remain preserved.

The first connected-target result is fixed131 versus Gaussian105 graph passes
out of512, but independent continuation gives93 versus91. Disconnection falls
192 to162 and243 to194. Node/pair first-run graph counts121/111 do not beat fixed.
Node's mean E+ difference versus Gaussian is-0.624 eV, conditional interval
[-1.084,-0.144]; versus fixed is-0.229[-0.739,0.269]. Energy is measured only for
one continuation and is mixed across conditions. No replicated neural increment
or equilibrium distribution is demonstrated.

Immediate research decision:

1. Retain the exact spatial source-law framework and fixed-tree reference.
   Do not scale the current pair network or tune variants against these outcomes.
   Better prior likelihood did not establish better final chemical graph support.
2. Diagnose one specific source-fit/output-utility mismatch using training-only
   diagnostics. A covariance/scale-matched Gaussian is a missing control before
   attributing fragmentation changes specifically to topology. Avoid several
   architectures or a large sweep without a concrete mechanism.
3. Freeze the next bounded comparison before new generation. Include fixed tree
   and Gaussian, original charge/spin and every condition. Separate training-seed
   replication from more draws of one fitted model. No follow-up protocol is
   declared frozen yet; additional physical queries are not queued.
4. Broader held-composition and learned-generator comparisons are still needed.
   The eight-case panel is development evidence. A new physical law and universal
   perfection are not prerequisites; support for the actual claim is. Neither
   low predicted energy nor a normalized source establishes Boltzmann matching.

Reproducible artifacts:

- `research/evidence/tree_prior_results_v1.json` / `.csv`: all studies and decisions.
- `research/evidence/generator_reference_registry_v2.json`: checkpoints and source laws; old registry retained.
- `research/evidence/tree_prior_confirmation_audit_v2.json` and `tree_prior_energy_audit_v2.json`: strengthened audits.
- `research/evidence/tree_prior_scheduler_v1.json`: source snapshots and terminal job records.
- `research/figures/tree_prior_v1/`: structural replication and energy PDF/PNG figures.
- `scripts/research/summarize_tree_prior.py`: rebuild tables, summary, registry and figures from audits.

Current sampling is midpoint64 at T1 plus0.025-A COM noise. Non-Gaussian model
flags require explicit source positions and a matching prior-density callback.
Do not use the independent-Gaussian velocity-score proxy with correlated FM
pairing. This pilot is FM-only; original BGFM hooks remain intact. Audits replay
stored source draws/readouts, not training trajectories or final neural generation,
and do not independently re-query the physical oracle.

Protected:722 reserved outcomes; old12/18 evaluated cohorts;2,560 earlier orbit
outputs;6,144 tree-study outputs. None may enter fitting. Preserve OMol25, bond
loss zero, max_atoms200, separate environments and no home writes. Earlier
collision-pairing, routing and latent-mass sweeps remain closed. User handles
submission; `paper/angular_working.tex` is still a diagnostic draft, not a
manuscript demonstrating the current tree-prior claim.
