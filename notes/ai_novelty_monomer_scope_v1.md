# AI contribution and evidence boundary at the monomer benchmark

There is a concrete AI-method candidate, not an established ICLR-level originality
or usefulness result. Its object is the generator's learned probability source,
rather than only an auxiliary loss. An added network is neither necessary nor
sufficient to establish an AI contribution.

| Component | Current implementation | Evidence and boundary |
|---|---|---|
| Bond-free spatial source | Composition/electronic-state-conditioned latent-tree mixture with normalized continuous edge laws | Requires no supplied chemical graph. Normalization and source sampling are tested; tree ensembles and their identities are prior art. |
| Learnable source | Node/pair networks fit exact marginal coordinate likelihood | Held source NLL improves. Extra final-generation utility has not been established in earlier panels. |
| Flow integration | Explicit source sampling, type/rotation coupling and source-law guards | Geometry gains repeat on broad development data. The final noisy flow has no qualified absolute density or Boltzmann claim. |
| Static tree context | Parameter-matched actual/sham residual adapters | Negative controlled result; it is retained but not promoted or scaled. |
| Prospective monomer test | Reference-qualified neutral singlet compositions, selected before generated outcomes | All21 references pass; exact composition overlap with two verified corpora is absent. The experiment is running, so no new method advantage is yet claimed. |

Closest primary sources:

- [Meila and Jaakkola,2000](https://publications.ri.cmu.edu/tractable-bayesian-learning-of-tree-belief-networks):tractable factored distributions and latent mixtures over trees.
- [Duan and Dunson,2023](https://www.jmlr.org/papers/v24/22-0252.html):spanning-tree likelihoods for dependence modeling.
- [ET-Flow,2024](https://arxiv.org/abs/2410.22388):harmonic molecular priors and equivariant flow matching with a supplied molecular graph.
- [Latent-CFM,2026](https://proceedings.mlr.press/v300/samaddar26a.html):flow conditioning on pretrained latent features. Generic latent/component conditioning is not a new invention here.

The candidate distinction is the explicit, learnable spatial source without
observed bonds, its symmetry-consistent integration with the molecular flow, and
whatever controlled molecular benefit it demonstrates. This focused comparison
is not an exhaustive originality certificate. Simple source engineering alone
may be insufficient for ICLR; the complete method needs a useful distinction,
meaningful baselines and reproducible evidence on a coherent task.

The monomer evaluation is deliberately narrower than unrestricted OMol25. It
does not alter old outcomes or use reference coordinates as generator inputs.
First-continuation node/pair comparisons are exploratory; the second continuation
only covers Gaussian/shell/harmonic controls. A positive learned-prior result must
be independently replicated before a repeated-learning-benefit claim.
