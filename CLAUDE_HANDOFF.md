# Claude continuation — useful source structure, next transport-context hypothesis

Checkout `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md`,
`research/TREE_MOMENT_STATE_20260914.json` and
`notes/tree_conditioned_transport_brief_v1.md` first.

The latest study is COMPLETE: four source methods on two matched continuations,
4,096 new outputs, every source draw and structural readout replayed. Jobs46364922_0/1
completed0:0 after21:11/21:06. Query Slurm before assuming later work is absent.
Unit Gaussian / covariance Gaussian / harmonic tree / shell tree graph passes/512
are103/106/112/136 and94/96/128/118. Shell minus covariance Gaussian differences are
+5.86 pp[2.15,9.38] and+4.30 pp[0.59,8.01]. Shell minus harmonic is+4.69 pp[0.98,8.40]
and-1.95 pp[-5.66,1.76]. All intervals are conditional on models and the fixed eight
compositions; they are not multiplicity-adjusted or broad generalization evidence.

The Gaussian/shell checkpoints are the same as the previous connected studies;
only their generation streams are new. Preserve the earlier weak93 vs91 comparison.
The result supports source structure beyond the tested single-Gaussian covariance,
but does not establish a repeatable narrow-shell advantage or an extra learned-
affinity increment. Covariance Gaussian uses a fixed128-tree estimate; harmonic
and shell conditional covariances match analytically. Calibration error is recorded.

Core implementation: `cfm_mol/tree_prior_controls.py`, with tests in
`tests/test_tree_prior_controls.py`; existing FM runner adds optional source controls.
Eleven tree/control tests pass. `audit_tree_moment_controls.py` checks all4,096
outputs, matched training rows, source laws and checkpoints; it does not retrain
optimizers or reproduce final neural integration. No physical query was added.
The preceding all-output energy result remains5,120 calls and one training seed.

The next concrete hypothesis is an equivariant transport adapter receiving the
sampled latent tree. Current training discards that tree after constructing x0;
the model never receives it. This is valid marginal FM, but possibly an avoidable
information bottleneck. The brief specifies actual/sham/no-context controls,
strict source-tree permutation bookkeeping and checkpoint guards. It is a design
only, NOT an implemented/validated method. Keep the same shell prior for the first
probe; do not change source learning and transport conditioning simultaneously.
The user authorizes bounded continuation without another permission question.

`paper/tree_working.tex` is the new internal manuscript with proofs and all current
results; `paper/tree_abstract.txt` is its truthful abstract. Official2027 styles were
verified byte-for-byte. It remains a development manuscript, not submission ready.
The older `angular_working.tex` and `main.tex` preserve their separate histories.

Protect722 reserved outcomes, old12/18 cohorts,2,560 orbit outputs and all10,240 tree
outputs from fitting. Preserve bond-free OMol25, max_atoms200, original charge/spin,
separate environments and no home writes. A tree is a latent dependence structure,
not a chemical bond or validity guarantee. The noisy midpoint64 T1 sampler has no
qualified final density, ESS or Boltzmann law. Old failed routing/collision/latent-
mass sweeps remain closed. ICLR goal active/unachieved; user handles submission.
Official deadlines: September18 abstract, September25 paper,23:59 AoE.
