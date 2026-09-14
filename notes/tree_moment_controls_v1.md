# Moment controls for the latent-tree source

Frozen question: is the observed fragmentation reduction explained by the source
covariance, by a mixture of tree-dependent Gaussian shapes, or by finite-length
non-Gaussian edge displacements? This experiment adds controls to identify the
mechanism. Neither control is a proposed new neural architecture.

For the lognormal-radius edge with mean length ell and log width s, isotropy gives
zero vector mean and Cartesian covariance v I, v=ell² exp(s²)/3. Let A_T map tree
edge displacements to centered positions. The coordinate covariance conditional
on T is C_T=A_T diag(v_e) A_T^T, independently in each spatial dimension.

The harmonic-tree control keeps exactly the same tree law and replaces each edge
displacement by N(0,v_e I). Thus all conditional and marginal second moments match
the original shell-edge tree exactly in ideal arithmetic. Its normalized spatial
density is the same matrix-tree ratio with Gaussian edge factors. This is a
mixture of Gaussians, not one Gaussian. Prior graph-informed harmonic FM methods
already exist; this control does not claim to reproduce their trained checkpoints.
See [ET-Flow](https://arxiv.org/abs/2410.22388) and the prior-art discussion in
`notes/tree_mixture_prior_v1.md`.

The covariance-Gaussian control removes the latent tree mixture. It uses one
centered Gaussian with a frozen estimate of E_T C_T. Calibration averages128
analytic tree covariances, using composition-specific fixed seeds and exact
weighted Cayley sampling for a_ij=u_i u_j. It uses no molecular coordinates,
training labels, energy or generated output. Exact averaging over same-species
permutations restores species symmetry despite finite calibration. The resulting
Gaussian is exactly normalized, while its covariance is an approximation to the
infinite mixture's covariance. No diagonal jitter or fitted scale is introduced.

An independent8,192-tree reference gives relative Frobenius differences2.4--6.1%
and trace differences at most3.5% across the eight development compositions.
This is a Monte Carlo diagnostic, not a rigorous error bound or a check of every
training composition. `research/evidence/tree_covariance_calibration_v1.json`
preserves every condition and calibration cost.

Two continuation/data-order seeds match the preceding connected-target studies.
Each new control receives the same3,000 selected training examples and update
budget as its corresponding frozen Gaussian/fixed-tree references. All four
methods receive fresh generation streams,64 outputs per condition, retaining
all eight development conditions. The total is4,096 new evaluation outputs and
zero new physical-oracle queries. No affinity network is fitted or scaled here.

Report each seed separately. Compare fixed shell tree against both new controls
and unit Gaussian for graph support, geometry support and fragmentation; retain
overlaps, exceptions, diversity and costs. Paired intervals condition on the fixed
eight compositions and fitted models. Two continuations do not establish new-
composition generalization or quantify pretraining-seed uncertainty.

If covariance alone matches the benefit, revise the topology-specific claim. If
the shell tree improves on covariance and harmonic controls, that supports its
non-Gaussian source structure; it still does not establish learned-affinity value.
Mixed results remain mixed. Do not drop metals, adjust graph perception, or tune
the new controls against outcomes. Only a subsequent concrete hypothesis can
justify another learned-source variant. Earlier6,144 tree-study outputs and
all protected cohorts stay outside fitting. The final noisy sampler remains
unqualified for absolute density, ESS or Boltzmann-law claims.

The official2027 schedule was checked on September14: abstract September18 and
paper September25, both23:59 AoE. Source:
[ICLR dates](https://iclr.cc/Conferences/2027/Dates).
The user handles authorship and actual submission. This schedule does not make
the current method or manuscript scientifically ready.
