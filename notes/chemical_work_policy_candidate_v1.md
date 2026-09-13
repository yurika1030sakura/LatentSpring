# Candidate: predict finite chemical work to choose useful edits

Implementation update (2026-09-13): `cfm_mol/chemical_work.py` now implements
shared passive-context paired 3D work, a matched graph-only ablation, and typed
linear bond energies. Both seeds have completed fixed400-step diagnostic and
full fitting, with no new physical labels. The held-parent 3D work error does
not beat the linear model. `cfm_mol/chemical_work_policy.py` implements exact
normalization over valid forward/reverse catalogues, including the uniform
mixture and actual map volume. Eight focused tests and all5,364 trained pair
reversals pass. No one-pass cache across different active masks is implemented.

The completed18-parent molecular evaluation uses498 raw oracle calls to score
all231 valid edit endpoints and18 sources;20 invalid actions are retained.
Forward policies are frozen before obtaining candidate energies. This estimates
exact one-step corrected utility, not a sampled trajectory or an inference
result after training/preparation costs. Linear bond energies currently give
higher point utility than both frozen3D models. See the latest state/summary for
full replay status and the final paired comparisons. The candidate description
below records the hypothesis before implementation; it is not an AI novelty
certificate.

The stronger controls close the current neural-mobility superiority claim.
At128 queries on the18 new parents, mean potential changes are-0.60935 eV for
collective learning,-0.59773 for the bare edit and-0.61205 for small root noise.
Learned-minus-root-noise is+0.00270 eV with a descriptive interval spanning zero.
The earlier advantage over physical arcs is real for that comparison but does
not isolate a useful learned contribution. Preserve every result; do not scale
the current mobility weights or present their arc comparison as AI superiority.

The observed remaining bottleneck is edit choice. In the bare-edit transfer
chains,576/614 joint attempts have valid scored geometry, but531/576 raise the
potential; only55 are accepted. Median proposed increase is0.41190 eV. Thus
geometrically reasonable edits still waste physical queries on unfavorable
chemical changes. This motivates a learning problem; it does not prove it is
learnable or novel enough for ICLR.

## Concrete AI hypothesis

Learn the finite physical energy change of a proposed chemical edit from its
3D environments, then use those predictions to choose the edit before the
expensive physical query. Keep the already effective reversible coordinate map.
The candidate representation is a shared masked passive context with paired
active-geometry queries: mask BOTH moved atoms, encode the unchanged surroundings,
and query the original and counterfactual placements through a shared conditional
energy decoder. Their difference predicts finite work and reverses sign when
the edit is reversed. The known restraint change and map Jacobian are explicit.

For terminal edits, passive relative coordinates and passive bonds are unchanged.
Center the passive context in its own frame; express both active placements in
that same frame. Do not accidentally expose the original active positions through
the context encoder. A one-pass passive message cache can subtract the two active
atoms' contributions for each candidate; the two-root decoder must retain their
joint interaction. This is a proposed representation, not implemented yet or an
established general energy model. Pairwise reversal consistency alone does not
prove global cycle consistency or physical accuracy.

The policy can enumerate geometrically valid reversible bare edits and use a
fixed informed weight such as log sigmoid of predicted dimensionless log target
ratio. Normalize over the ACTUAL candidate catalogue and add a uniform component
if needed. Recompute the reverse catalogue/probability at the accepted candidate.
The real MH correction includes the actual physical energy, intrinsic map volume
and forward/reverse selection probabilities. Never treat a learned ranking or
validity filter as an excuse to omit the proposal ratio.

## Prior art and the claim boundary

[Informed/locally balanced proposals](https://arxiv.org/abs/1711.07424),
[Local Self-Balancing MCMC](https://openreview.net/pdf?id=WEKfdiZYZi-), and
[discrete Langevin proposals](https://proceedings.mlr.press/v162/zhang22t/zhang22t.pdf)
already cover informed discrete sampling and learned/adapted proposal ideas.
[MOSAiCS](https://arxiv.org/abs/2307.15563) already explores molecular graphs by
Monte Carlo with quantum-property objectives. None of these general principles
is a new contribution here. The targeted primary-source search is not an
exhaustive novelty certificate.

The candidate distinction must be a useful, efficient 3D finite-work representation
for reversible chemical transports, with original electronic states and explicit
coordinate measure. Test it against uniform catalogue selection, cached-force
linear work, a linear bond-energy model and a graph-only learned model. A generic
GNN wrapped in standard MH or a better training loss is not enough by itself.

## Immediate bounded experiment

`scripts/research/chemical_work_catalogue.py` enumerates all492 eligible bare
edits on36 original FIT sources.447 pass the existing geometry/reverse checks;
all45 failures remain. Scoring36 sources plus447 candidates in both inversion
orientations requires966 raw calls. No evaluated or reserved parent is fitted.
The plan and labels are frozen/audited separately. This is training information,
not an oracle-assisted deployed sampler or evidence of an optimal policy.

After the labels are complete, implement the compact masked-context work model
and the linear/graph-only controls, train on parent-balanced objectives, and
quickly test corrected molecular proposals at matched budgets. Retain existing
18-parent and12-parent evaluations as development cohorts; do not relabel repeated
testing as untouched final evaluation. Keep722 reserved outcomes unqueried.
Do not resume optimizer cleanup or demand perfect source geometries first.
