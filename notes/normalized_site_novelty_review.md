# Normalized site guide: contribution and prior-art boundary

September12, 2026. This is a candidate AI method, not an established ICLR novelty
or superiority claim. The current implementation changes the conditional density
architecture and sampler; it is more than adding another penalty to FlowMol.

## Actual learned object

`NormalizedSiteGuide` predicts a normalized directional law. Continuous atomic
number/radius descriptors, masked geometry, graph and electronic context feed two
message blocks. A learned equivariant center corrects the coordination-site prior;
paired offsets indexed by passive atoms and invariant mixture weights define a
finite vMF mixture. Its density and spherical score are both explicit. The vector
ablation retains only the corrected center and is currently at least as good on
training force fit. More components are not themselves a demonstrated benefit.

Two roots, radii and directions are generated autoregressively after a proposed
attachment exchange. The implementation mixes the WHOLE learned joint proposal
with a physical-site proposal and evaluates both complete densities in each
direction. It retains the source-dependent reverse templates, r^-3 Cartesian
factors, action counts, fair root order and single-draw support rejection.
This does not need the unknown likelihood of the frozen FM initializer.

## Closest verified sources

| Primary source | Established overlap | Difference requiring evidence here |
|---|---|---|
| [VonMisesNet, ICML2023](https://proceedings.mlr.press/v202/swanson23a.html) | GNN-predicted von Mises mixtures for molecular torsions and Boltzmann-oriented conformation generation. Full primary paper was inspected. | We model directions on S2 after attachment edits, with full coordinate/reverse correction and off-equilibrium force examples. This distinction alone does not establish useful novelty. |
| [MARS,2021](https://arxiv.org/abs/2103.10432) | Molecular graph editing with learned GNN proposals and MCMC. Primary abstract verified. | Here the coordinate proposal and its measure are part of the target-preserving edit. Graph-editing MCMC itself is not new. |
| [Equivariant VFM, ICML2025](https://proceedings.mlr.press/v267/eijkelboom25a.html) | Symmetry-aware molecular generation and post-hoc control of unconditional models. Primary proceedings metadata/abstract verified; PDF fetch failed. | The current correction is a learned MH proposal for an explicit restrained physical target. No claim of a new general post-hoc-control principle. |

Timewarp, Markovian flow matching, masked-conditionals-as-MH and MOSAiCS remain
in the manuscript's related work. Configurational-bias/regrowth Monte Carlo and
directional distributions are longstanding physical/statistical foundations;
neither geometry regrowth nor density/Jacobian correction should be presented as
a new general idea. A targeted search is not an exhaustive novelty certificate.

## What the current evidence supports

- Both normalized learned variants finish the diagnostic graph conversions in
  both256-step repetitions. Eight complete trajectories and all random streams
  replay; independent calculations verify the FULL marginal mixtures.
- Training MSE falls from the site's708.1 to vector558.8/545.6 and
  mixture571.6/549.6. These are pointwise TRAINING force errors.
- The17-atom screen gives learned-mixture99/128 and86/128 valid candidates,
  compared with site101/128 and96/128 and initialization93/128 and94/128.
  The earlier zero-validity failure is absent, but the physical component and
  initialization explain much of the support; learned transfer superiority is
  not established.
- The inherited preparation cost and absent equilibrium/generalization evidence
  still prevent an ICLR-readiness claim. The previous fully cost-matched site
  controls remain stronger evidence than new short-step comparisons alone.

The candidate contribution must be a demonstrably useful combination for changing
chemical connectivity together with geometry under a specified physical target.
We still need cheaper training/preparation or measured amortization, stronger
independent-condition results, and actual distribution qualification. Generic
mixtures, equivariance, force fitting, defensive MH and correct math are necessary
ingredients, not sufficient novelty claims.

## September12 follow-up: accepted movement and newer dynamics models

The full Timewarp preprint was downloaded and its training section inspected.
It already refines proposals through an acceptance-ratio objective together with
likelihood and entropy terms (equations15--17). Optimizing MH acceptance is not
a new general learning principle. L2HMC already optimizes an expected squared
jump criterion. These remain relevant precedents if pointwise force fitting is
replaced by an objective closer to accepted movement.

TITO, published in Science Advances in2026, learns lagged molecular transition
distributions with conditional equivariant flow matching across compositions
and time lags. Its preprint method section was inspected, not just its title.
Broad transferable flow-based dynamics is therefore not a distinctive claim
for this project. A possible distinction here is reversible connectivity edits
with explicit coordinate probabilities and an electronic-state-conditioned
physical target; it still requires useful method and benchmark evidence.

Primary sources: [Timewarp](https://arxiv.org/abs/2302.01170),
[L2HMC](https://arxiv.org/abs/1711.09268),
[TITO preprint](https://arxiv.org/abs/2510.07589),
[TITO publication record](https://pubmed.ncbi.nlm.nih.gov/41950332/).
File hashes and exact inspected scope are recorded in
`research/evidence/accepted_move_prior_art_20260912.json`.
