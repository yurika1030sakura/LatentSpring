# Prior-art boundary for retaining source-tree context

Checked against primary publications on September14,2026. This is a focused
comparison, not an originality certificate.

[Latent-CFM, Samaddar et al., AISTATS2026](https://proceedings.mlr.press/v300/samaddar26a.html)
conditions flow matching on features from pretrained latent-variable models to
exploit target clustering. Its GMM variant supplies cluster membership to the
vector field. Thus latent or mixture-component conditioning, reduced regression
ambiguity and data-efficiency motivation cannot be claimed as our general inventions.

Our implemented variant supplies an auxiliary tree sampled from the explicit
coordinate source, without a pretrained target encoder or observed chemical graph.
The tree has variable molecular size, follows source atom relabeling through
the symmetry coupling, and is encoded through invariant pair features in an
equivariant coordinate field. These are implementation/task distinctions. Whether
they yield a useful and sufficiently distinctive method requires the actual/sham/
no-context molecular experiment and later transfer evidence.

[ET-Flow](https://arxiv.org/abs/2410.22388) uses graph-informed harmonic priors for
conformer generation. The harmonic source idea is prior work; our no-bond setting
and latent source graph should be distinguished from a supplied chemical graph.
[SemlaFlow](https://openreview.net/pdf/c1b124c4e1cff3f1c4c21d3a99934425394ad173.pdf)
already uses latent graph attention for efficient molecular generation. Generic
graph attention or adding edge features is not an adequate originality claim.

The main empirical question is specific: does retaining the sampled source
dependence structure improve a bond-free molecular flow beyond an equally sized
network receiving an independent tree and beyond the same source without context?
The framework must be assessed as a whole, including its explicit source law;
the small residual adapter alone does not certify ICLR novelty. There is no claim
of a new physics law or of a qualified Boltzmann sampler.
