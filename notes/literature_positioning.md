# Prior work and the remaining contribution test

- [LDR, ICML 2026, final arXiv version](https://arxiv.org/pdf/2602.03729)
  already combines off-policy energy-labelled log-dispersion with data-based
  training. Its molecular backbones are discrete normalizing flows, including
  internal-coordinate spline flows and Cartesian TarFlow. The energy variance
  idea is not our novelty; a CNF implementation must justify its numerical and
  computational costs against that setting.
- [Verlet Flows](https://arxiv.org/pdf/2405.02805) studies a flow construction
  with tractable numerical density evaluation and discusses why stochastic
  trace variance is unsuitable for exact importance reweighting.
- [Flow Perturbation++](https://arxiv.org/html/2601.21177v1) addresses Jacobian
  estimation and SMC reweighting with frozen pretrained models, including
  Gaussian and Rademacher probe baselines. Its target quantity and inference
  setting differ from the expected squared training residual studied here.
- [Finlay et al., ICML 2020](https://proceedings.mlr.press/v119/finlay20a.html)
  already uses Jacobian and kinetic regularization to simplify neural ODE
  dynamics and reduce integration cost. Our smooth-geometry trial is an
  engineering response to measured stiffness, not a claimed new principle.
- [Koehler, Klein and Noe, ICML 2020](https://proceedings.mlr.press/v119/kohler20a.html)
  already constructs equivariant pair-kernel flows with analytic divergence.
  The new restricted radial reference follows that architectural class; its
  value is as an exact-trace control, not as an architectural novelty claim.

A defensible contribution would require a reproducible, computationally useful
way to train a clearly defined multi-composition molecular flow with noisy
density estimates, plus physical improvements under strong matched controls.
The independent-product identity, a conditional factorization, corrected code
or the present constructed toys alone do not satisfy that contribution test.

## Additional baseline check (September 9)

[EWFM v2](https://arxiv.org/html/2509.03726v2) uses importance-weighted
conditional FM with proposal-density correction and amortized sample buffers.
A softmax of energy labels alone on the archived perturbations would instead
train an energy-tilted proposal; it must not be presented as a faithful EWFM
implementation. A proper comparison must specify the proposal and target
domain, count density/buffer costs and report finite-sample weight degeneracy.

## Regression-trained exact-likelihood flows

[RegFlow](https://arxiv.org/abs/2506.01158), accepted at ICLR 2026 according to
its [official repository](https://github.com/danyalrehman/RegFlow), trains
normalizing flows by regressing coupled pairs supplied by a pretrained CNF or
an invertible optimal-transport mapping, with a forward/backward consistency
regularizer. Thus replacing a costly CNF likelihood by a regression-trained
invertible student is an existing strong baseline, not a new contribution.
It is distinct from FALCON's few-step flow-map approach and must be considered
before committing to a redesigned exact-likelihood geometry architecture.

A possible subsequent physics-teacher study would optimize mean generalized
work using differentiable Gaussian paths and external energy/force evaluations.
This is a path-space KL / stochastic-normalizing-flow objective, also existing
prior work. It could train the forward and auxiliary backward proposals rather
than relying on fixed backward drift or hand-constructed templates. It has not
been implemented or shown useful in this project; do not report it as a result.


## Diffusion-bridge loss comparison, checked September 10, 2026

Sanokowski et al., Rethinking Losses for Diffusion Bridge Samplers, NeurIPS 2025,
https://arxiv.org/abs/2506.10982, is direct prior art for comparing on-policy LV
and reverse-KL training with learnable forward/backward processes. Our
forward-only score/pathwise expectation relation does not equate joint training
objectives. Choosing lower-variance gradients or distinguishing these objectives
is not sufficient novelty. The completed eight-atom LV failure and frozen
molecular gradient-noise measurement are local evidence, not a new general
critique of diffusion bridges.
