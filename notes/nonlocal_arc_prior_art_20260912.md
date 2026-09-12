# Prior-art boundary for conditional nonlocal arc proposals

Checked September12,2026 against the primary sources below. This is a targeted
positioning check, not an exhaustive novelty search. A useful complete-chain
result and representative comparisons are still required.

- Classical configurational-bias Monte Carlo already grows molecular segments
  with biased placement and corrects that bias on acceptance. Its authors'
  [original-paper record](https://siepmann.chem.umn.edu/publications/3) and the
  [Towhee implementation theory](https://towhee.sourceforge.net/algorithm/cbmc.html)
  document this family. Feasibility-aware regrowth, cheap biased placement and
  reverse reconstruction cannot be claimed as new general ideas here.
- [Self-learning Monte Carlo](https://arxiv.org/abs/1610.03137) learns an effective
  update from preliminary simulations. Its
  [deep-neural extension](https://arxiv.org/abs/1801.01127) learns effective models
  for efficient proposals. A neural energy surrogate followed by exact correction
  is therefore not standalone AI novelty.
- Monroe and Shen's
  [Learning Efficient, Collective Monte Carlo Moves with Variational Autoencoders](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=932532)
  uses learned encoders/decoders with explicit probabilities, and discusses
  autoregressive decoding and configurational-bias trial selection. Tractable
  learned proposal densities and physics-informed decoders also have precedent.
- [Geodesic Slice Sampling on the Sphere](https://jmlr.org/papers/v26/23-1158.html)
  is direct prior art for generic geodesic MCMC. Its guarantees do not automatically
  cover our graph-changing, support-restricted joint kernel.
- [NHMC](https://arxiv.org/abs/2607.15682v2) learns stochastic Hamiltonian paths and
  uses recorded nonequilibrium work for statistical correction. Its August26,2026
  abstract explicitly discusses path overlap failures and a molecular feasibility
  study. It overlaps the old work-correction motivation; it does not establish
  our molecular efficiency or supply novelty for reusing path-ratio identities.
- [Graph Energy Matching](https://arxiv.org/abs/2603.23398v3) learns an invariant
  discrete graph energy for transport and local refinement. Based on its abstract,
  it is adjacent to graph-energy generation; a detailed architecture/benchmark
  comparison remains necessary before asserting a specific distinction.

The present candidate couples a learned conditional placement-energy score to
analytic feasible-circle intervals, exact probabilities of their implemented
interpolants, two root/radius regeneration and marginalization over decoder
orders. It targets joint constitutional connectivity and3D geometry with a
separate expensive physical oracle. This is a concrete algorithmic candidate;
none of its generic ingredients or the new Huber force/work objective alone
establishes novelty. Its useful coupling must be shown empirically and separated
from simply fitting a cheap potential.

After the first frozen learned-chain test, the most informative additional
comparisons are a classical biased-regrowth/multiple-trial physical control and
a same-score local proposal (to separate conditional prediction from the arc
construction), alongside appropriate learned molecular generator/MCMC baselines.
Do not present a failed repurposed EACF refiner as native EACF performance. Do not
claim every graph generator is directly comparable under a different physical
target. Declared task, data, support and costs must make each comparison meaningful.
