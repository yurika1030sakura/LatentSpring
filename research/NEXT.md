# Next — source structure supported; test how the flow uses connection information

Read `research/TREE_MOMENT_STATE_20260914.json`,
`notes/tree_moment_controls_v1.md` and
`notes/tree_conditioned_transport_brief_v1.md`.

The new control study is complete and audited. Unit Gaussian / covariance Gaussian /
harmonic tree / shell tree graph passes out of512 are103/106/112/136 for the first
continuation and94/96/128/118 for the second. Shell versus either single Gaussian
has positive conditional paired intervals in both; shell versus harmonic changes
sign. The source framework has a useful signal beyond the tested covariance
control, while an extra neural affinity increment remains unproved. The covariance
is a fixed Monte Carlo approximation; all intervals condition on fitted models
and eight development compositions, without multiplicity adjustment.

The Gaussian/shell references reuse the same two earlier checkpoints with fresh
generation streams. Preserve the original131 vs105 and93 vs91 streams; do not call
the new samples additional independent fits. No source should be selected separately
by seed. Source structure, narrow radial shells and affinity learning are distinct
claims. Only the first has strengthened evidence in this experiment.

Next authorized work:

1. Implement the single architecture hypothesis in the transport brief: retain
   the sampled tree as context in the flow instead of supplying only its coordinates.
   Use a runtime edge-feature adapter, preserve the original FlowMol source and
   bond-loss zero, and correctly permute the tree with source atom assignments.
2. Test zero-initialization equivalence, joint symmetry bookkeeping, missing-context
   guards, nonzero adapter gradients and checkpoint restoration. This is not a
   request for another broad redesign or parameter sweep.
3. Freeze a bounded actual-tree/sham-tree/no-context molecular comparison with the
   same source, backbone and data. Actual/sham parameter counts must match. The
   architecture has not been implemented or qualified yet; do not report its
   regression-risk motivation as an achieved generation gain.
4. If it helps, proceed to a separately frozen composition-generalization panel and
   learned-generator comparison. Do not tune on or fit any past evaluation outputs.
   A more complex network or better source NLL alone is insufficient evidence.

Current artifacts:

- `research/evidence/tree_moment_audit_v1.json`: all4,096 new source/outcome replays,
  per-composition results, all controls, checkpoints and conditional uncertainty.
- `research/evidence/tree_moment_results_v1.csv`: complete aggregate table.
- `research/figures/tree_moment_v1/`: publication-format PDF/PNG comparison.
- `research/evidence/tree_moment_scheduler_v1.json`: both46364922 tasks completed0:0.
- `paper/tree_working.tex`, `paper/tree_refs.bib`, `paper/tree_abstract.txt`: current
  internal manuscript and abstract. The moment results are generated from the audit.
- `research/evidence/tree_paper_template_check_v1.json`: local2027 style/bibliography
  files match the official archive. This is formatting evidence, not readiness.

Eleven tree/control tests pass, including density, covariance, sampling and symmetry.
The covariance calibration independently replays. Audits do not rerun optimizers or
final neural generation. There are zero new physical queries in this control study;
5,120 were used by the earlier energy readout. No tree job remains active at the
recorded check; verify Slurm before starting/recovering work.

Protected:722 reserved outcomes; old12/18 evaluated cohorts;2,560 orbit outputs;
all10,240 tree-study outputs. None may enter fitting. Keep OMol25 primary, max_atoms200,
original charge/spin, separate environments and no home writes. The final noisy
midpoint sampler has no qualified absolute density, ESS or Boltzmann distribution.
Old routing/collision/latent-mass sweeps stay closed. The ICLR goal remains active
and unachieved; the user handles authorship/submission. Abstract deadline is
September18 and full paper September25,23:59 AoE, checked against official ICLR2027.
