# Next hypothesis: learn the source from generated-sample utility

Status update: the design is implemented and the bounded pilot is running; see
`notes/source_utility_method_v1.md` and `research/SOURCE_UTILITY_STATE_20260914.json`.
No performance gain is claimed. The remaining text records the original design brief. The completed monomer test
does not establish an extra benefit of coordinate-likelihood source fitting.

## What the new benchmark resolves

All21 prospective neutral organic references pass the unchanged assay, and exact
composition matches are absent from the two checksum-verified processed corpora.
The first continuation's node/pair source models still do not beat the fixed
shell source. This removes poor reference-assay coverage as a sufficient explanation
for their lack of gain. It does not identify a unique cause: capacity, optimization,
data coverage and the source-fitting objective remain possible factors.

One specific testable change is to fit source probabilities to terminal generation
utility while keeping the generator fixed, instead of fitting source likelihood
to original molecular coordinates. Avoid adding another context/attention module
at the same time. The existing static context recipe stays closed to scale-up.

## Exact source-space objective for a fixed generator

For a fixed condition c, let q0(x0|c) be the normalized base source and K_theta(y|x0,c)
the FROZEN actual generation kernel, including its finite integrator and terminal
noise. Let q_eta be a candidate source with compatible support. Then for bounded
utility R(y,c),

    J(eta,c) = E_{q_eta K_theta}[R]
             = E_{q0 K_theta}[(q_eta(x0|c)/q0(x0|c)) R(y,c)].

The ratio is evaluated at the recorded source coordinates. This identity does
not require an inverse generator or its marginal output density. It does not
license interpreting this ratio as the marginal output-density ratio. A cached
training bank can therefore provide a differentiable importance objective using
the exact matrix-tree source marginal; no gradient through discrete tree sampling
or the graph-validity classifier is required.

For a fixed eta this is an importance identity. Optimizing a finite cached bank
can overfit, so fresh validation generations are essential. Source weights/ESS
are source-space diagnostics and do not certify independent target samples or
a thermal ensemble. Every attempted output, including failures, must contribute
to the declared utility; no hidden rejection or censoring is permitted.

## Analytic control of distribution change

If only the tree affinities change and the normalized edge kernels remain fixed,
the Markov chain is T -> X0 -> Y with common kernels. Standard data processing gives

    KL(p_eta(Y|c) || p0(Y|c))
      <= KL(q_eta(X0|c) || q0(X0|c))
      <= KL(p_eta(T|c) || p0(T|c)).

The tree divergence has an exact partition-function expression:

    KL_tree = sum_e m_eta,e (log a_eta,e - log a0,e)
              - log tau(a_eta) + log tau(a0),

where m_eta,e is the prior tree edge-inclusion marginal. Thus an input-side
regularizer or independently verified per-condition budget can bound the change
of the raw output law without claiming a tractable marginal output density.
This statement requires the SAME frozen generation kernel; it does not apply to
comparisons in which the flow weights also change, such as the completed source-
and-flow continuation studies. It applies to raw attempts, not automatically to
their rejection-conditioned distribution.

A possible conservative deployment construction mixes tree laws:

    p_mix(T|c) = (1-epsilon(c)) p0(T|c) + epsilon(c) p_eta(T|c),

with epsilon chosen so epsilon*KL_tree <= delta. KL convexity and data processing
then bound the output KL by delta. The corresponding coordinate density is the
same two-component mixture of known source densities, so sampling and evaluation
remain explicit. This is an optional construction to evaluate, not an implemented
algorithm. KL control alone does not bound every importance weight or guarantee
improved molecular diversity/validity; those must be measured.

## Bounded first experiment and failure conditions

Use one frozen shell-source flow per continuation. Select new FIT compositions
from the actual training corpus under the coherent monomer criteria, disjoint
from all declared evaluation panels. Generate a bounded source/output bank and
fit an equal-sized source head using actual utility versus within-composition
shuffled utility, with the original fixed prior as the third control. Source
coordinates and terminal outputs from previous evaluation campaigns may not enter
that bank. Keep kernel widths, flow weights and the initial source family fixed
for the first probe. Preparation and fresh-validation generation costs count.

Prespecify trust control, learning budget, validation streams and failure criteria
before fitting or evaluation. An offline importance gain alone is not a successful
generator. The decisive result is a fresh actual-utility advantage over both
shuffled-utility and unadapted controls, with diversity and cost retained. Do not
silently relax the trust budget or grow source-family capacity after looking at
the validation outcomes.

Importance sampling, reward-weighted fitting, latent/noise adaptation, KL convexity
and data processing are prior art; the identities above are not new theorems or
new physics laws. Originality would need to lie in a useful, well-supported source
adaptation method for this molecular setting. A dedicated related-work check is
still required before an originality claim. Optimizing an energy-related utility
would not by itself produce a Boltzmann law; a quality tilt of a generator is a
different target from a canonical ensemble.
