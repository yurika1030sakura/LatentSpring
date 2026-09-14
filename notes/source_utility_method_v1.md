# Frozen-decoder utility adaptation: bounded pilot v1

This implements `source_utility_learning_brief_v1.md`; the completed pilot fails its prespecified development gate. See
`notes/source_utility_results_20260914.md`; the text below records the frozen method. The base is each of the two frozen
shell-source flow continuations. No decoder, edge kernel or radial width changes.

The existing pair head predicts symmetric latent-tree affinities. Its exact
coordinate marginal integrates all spanning trees. A bank of base-source draws
and terminal outcomes supplies marginal source importance ratios. The objective
is the estimated utility improvement minus0.01 KL(tree_candidate||tree_base).
Utility is0.8 graph support plus0.2 geometric support, including all failures.
With M iid bank draws, M/(M-1) times mean[w(R-mean R)] estimates the fixed-model
utility difference without the same-sample centering bias. Optimization on that
bank can still overfit; this is not an unbiased fitted-model performance estimate.

Every deployed/trained source is a mixture with epsilon=min(1,0.25/KL_tree).
The identical edge kernels and frozen decoder give an upper bound0.25 nats on
raw-output KL relative to the unadapted source. This standard convexity/data
processing bound says nothing about Boltzmann correctness, output ESS, guaranteed
validity or retained graph diversity. It does not apply to older changed-decoder
comparisons. Effective-resistance marginals implement the exact tree KL and its
derivatives. Positive star-mesh elimination evaluates batched coordinate density.

The frozen protocol is `research/evidence/source_utility_protocol_v1.json`.
Selection uses a new seeded permutation of the checksum-verified real training
corpus, neutral multiplicity1 C/H organic compositions8-40 atoms, reference-assay
qualification and explicit exclusion of all official candidate/panel compositions.
There are24 FIT compositions and12 source-head held-out compositions, balanced
across three size bins. **Both originate from the flow training corpus**: the
held-out claim refers only to new source-head fitting. No global model-pretraining
or trajectory-disjointness claim follows. No previous evaluation coordinates,
generated samples or reserved outcomes enter FIT. Selection records preserve all
53 reference assays, including failures;36 pass and are selected.

Each frozen decoder generates64 base draws per FIT composition (1536). Three
same-capacity pair heads receive30 epochs, Adam0.003, clipping5, identical seeds
and condition order: real utility, one fixed within-composition utility shuffle,
and coordinate NLL on the24 FIT reference geometries. All use the same mixture
budget and tree-KL penalty. The NLL comparator matches head/update budget, not a
large-coordinate-data training budget. No early stopping or validation tuning.

Fresh64 draws per12 held composition and four sources (including fixed) give3072
validation outputs per decoder; the two-decoder campaign totals9216 attempts.
Report graph/geometry/utility, distinct connectivity per attempted draw, both
paired-draw and size-stratified composition intervals, all costs and failures.
The prespecified development gate requires real-utility point gains over both
fixed and shuffle in both decoders, pooled paired95 lower bounds positive for
both comparisons, and no greater than2pp pooled loss in distinct connectivity
per attempted draw versus fixed. A gate pass is still not submission readiness.
No sweep, budget increase or modified utility in response to these outcomes.

## Closest checked prior work and claim boundary

- Miao et al., *A Minimalist Method for Fine-tuning Text-to-Image Diffusion Models*
  (2025), https://arxiv.org/abs/2506.12036. Noise PPO freezes the pretrained
  diffusion model and learns a prompt-conditioned initial noise generator from
  reward. Therefore frozen-decoder/noise-distribution learning is not our novelty.
- Tang et al., *Inference-Time Alignment of Diffusion Models with Direct Noise
  Optimization* (2024), https://arxiv.org/abs/2405.18881. Optimizing sampling noise
  for reward, including treatment of nondifferentiable rewards and probability
  regularization, is also prior art. This work primarily adapts at inference time.
- Meila and Jaakkola, *Tractable Bayesian Learning of Tree Belief Networks* (2000),
  https://publications.ri.cmu.edu/tractable-bayesian-learning-of-tree-belief-networks.
  Tree-ensemble partition functions/marginals are established tools.

The narrower candidate contribution is useful bond-free molecular adaptation
through an explicit normalized structured source, with marginal importance
training and an exact tree-distribution trust calculation. This combination needs
molecular evidence and a fuller literature comparison; it is not yet established
as an original ICLR contribution. Generic importance sampling, KL convexity,
data processing, or a newly instantiated network are not original theorems.


Further primary-source check (same pilot; no protocol change):

- Wang, Harting, Barreau, Zavlanos and Johansson, *Source-Guided Flow Matching*,
  ICLR2026, https://openreview.net/forum?id=p56ZAQUCUr,
  https://arxiv.org/abs/2508.14807. Guidance by modifying the source under a frozen
  vector field, including source-space importance and MCMC sampling, predates this
  pilot. Its exact-target statement assumes the appropriate tilted source and
  transport; our bounded utility optimization makes no analogous target-law claim.
- Kim et al., *Better Source, Better Flow* (2026),
  https://arxiv.org/abs/2602.05951, learns conditional source distributions under
  flow matching and addresses collapse/stability. Conditional source learning
  alone is therefore not a distinct contribution either.
- Everink, *Random spanning tree Markov random field priors for Bayesian inverse
  problems in imaging* (2026), https://arxiv.org/abs/2605.18619, combines random
  spanning-tree connectivity with continuous pixel-difference priors. Random-tree
  spatial difference priors are not new in general.
- Zhou et al., *Guiding Diffusion Models with Reinforcement Learning for Stable
  Molecule Generation* (2025), https://arxiv.org/abs/2508.16521, fine-tunes molecular
  diffusion using physical rewards on QM9/GEOM. Reward-guided molecular generation
  is established, though that is a different task/training setup from this pilot.

A positive actual/shuffled/fixed test would be an internal mechanism result,
not a completed comparison with these methods. Structured marginalization, molecular
utility, adaptation cost and appropriate baselines would still need to support the
specific contribution. These findings narrow the originality claim; they do not
change the frozen experiment or justify declaring its outcome in advance.
