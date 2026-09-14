Current follow-up: the specified generator-calibration candidate is now implemented and audited. Read `notes/latent_mass_calibration_v1.md` and `research/LATENT_MASS_STATE_20260913.json`. It supplies a constructed-target coupling signal, not established distinctive neural or molecular-generator superiority. Earlier open-design text below is historical.

# AI contribution boundary after the complete-chain pilot

The cooperative architecture is implemented and correct at the tested interface,
but the complete-chain comparison does not establish its utility over the strong
single-edit controls. Engineering correctness is not an AI novelty result.
Likewise, a small improvement from fitting typed bond energies does not establish
a new general learned-sampling algorithm.

The following primary sources were refreshed on September13,2026. This is a
targeted overlap check, not an exhaustive novelty review.

| Proposed general claim | Prior work and implication |
|---|---|
| Learn an effective energy/update model, then correct Monte Carlo proposals | [Self-Learning Monte Carlo](https://arxiv.org/abs/1610.03137) already learns updates from trial simulations. Its [cumulative-update extension](https://arxiv.org/abs/1611.09364) proposes global changes through an effective model with statistically exact correction. This general pattern cannot be our novelty. |
| Add a neural network to that learned sampler | [SLMC with deep neural networks](https://arxiv.org/abs/1801.01127) already does this. A deeper scorer is insufficient. |
| Compose local edits into longer paths | [Path Auxiliary Proposal, ICLR2022](https://openreview.net/pdf?id=JSR-YDImK95) is a direct precedent, including energy-linearized proposals. Retained panels and exact MH are correctness mechanisms, not new general principles. |
| Local scores can miss mode proportions | [Wenliang and Kanagawa](https://arxiv.org/abs/2008.10087) already identify failures for isolated components and their mixing weights. This is a known issue, not a discovery made by our molecular pilot. |
| Repair score-based energy calibration or mode blindness with a new objective | [Towards Healing the Blindness of Score Matching](https://arxiv.org/abs/2209.07396) proposes modified divergences. [Diffusive Classification, ICML2026](https://arxiv.org/abs/2601.21025) learns energy models via noise-level classification and tests Boltzmann-generator use. Any calibration-based redesign must be distinguished explicitly. |

For exactly disjoint components, a conditional score can be invariant to their
mixture weights. This elementary argument must not be applied indiscriminately
to the entire FlowMol/BGFM training objective: data flow matching supplies other
information, and positive Gaussian noise connects supports. Practical blindness
and an exact disconnected-support theorem are different statements. The present
short molecular chains have no qualified reference component populations, so
they do not diagnose a mode-weight failure or validate a proposed cure.

Similarly, finite-work or free-energy reweighting cannot silently assume that an
arbitrary FM endpoint is an equilibrium starting distribution or has a known
likelihood. The existing source-density limitation remains. Current kernels use
actual endpoint energies and explicit forward/reverse proposals; query-stopped
outputs are not thereby equilibrium samples.

The useful work to retain is the audited target/interface, valid chemical maps,
frozen baselines, honest cost accounting and complete experimental records. A
future ICLR claim must identify a specific learned mechanism, explain its
distinction from these precedents, and demonstrate its benefit beyond the exact
zero-learning and strong learned/physical controls. Universal success across
all cases and a new physics law are not prerequisites; an unsupported novelty
claim is also not a substitute for those experiments.

Do not relabel the current failed cooperative model, generic residual learning,
or a standard energy-guided MH wrapper as a newly successful framework. Choose
one bounded main-method redesign before further architecture sweeps or paper
rewriting. No replacement method is implemented or validated by this note.
