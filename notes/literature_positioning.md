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
