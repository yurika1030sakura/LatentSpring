# Current contribution and physics boundary

The demonstrated candidate is a fixed random-tree harmonic coordinate source
combined with geometry-only self-conditioned molecular flow matching. Source
learning, a new neural architecture, a new matrix-tree theorem, and a new physical
fluctuation law are not demonstrated contributions.

| Prior work | Established ingredient | Proposed task-specific distinction |
|---|---|---|
| [ET-Flow](https://arxiv.org/abs/2410.22388) | Harmonic molecular source conditioned on a supplied molecular graph | Generate coordinates from composition without supplied chemical bonds; average over latent spatial trees. |
| [FlowMol3](https://arxiv.org/abs/2508.12629) | Molecular flow backbone and self-conditioning | Source replacement in an explicitly clamped coordinate task; this does not reproduce the published joint-generation benchmark. |
| [Everink, random spanning-tree MRFs](https://arxiv.org/abs/2605.18619) | Random-tree difference priors, Gaussian mixtures and posterior tree conjugacy | Intrinsic centered three-dimensional coordinates, species scales and a molecular FM application. Tree marginalization itself is prior art. |
| [Source-Guided Flow Matching](https://openreview.net/forum?id=p56ZAQUCUr) and [Better Source, Better Flow](https://arxiv.org/abs/2602.05951) | Adapting or learning source distributions for flow models | A specific explicit bond-free spatial source; generic source adaptation is not a novelty claim. |

Physics is applied through Gaussian spring energies, covalent-radius fluctuation
scales, translation removal and molecular symmetry. These are auxiliary source
energies and heuristic propensities, not electronic energies or a new physical law.
The selected harmonic edges have zero-rest-displacement Gaussian energies; atomic
radii set variance, not an enforced bond length or excluded-volume guarantee.

Jarzynski remains in the manuscript as a qualified connection, not evidence for
the reported validity gains. [Jarzynski's nonequilibrium identity](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.2690),
[Hummer--Szabo pulling reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC31107/)
and [escorted free-energy simulations](https://arxiv.org/abs/0804.3055) motivate
energy correction with complete work/Jacobian or path-law accounting. The appendix
specifies a finite-normalizer target and exact invertible map, and shows the
normalizer identity by change of variables. Current noisy numerical samples have
not undergone that qualified correction. No Boltzmann law or ESS result follows.

Evidence for utility: harmonic versus isotropic Gaussian improves on the12-case
development panel and the frozen10-case additional panel. The latter is limited
to17-28-atom neutral singlet organic compositions and the same two model seeds.
The new covariance-Gaussian control tests attribution beyond second moments.
The original endpoint checkpoint comparison tests a separate task-adapted native
baseline; different training histories must be disclosed. Neither result alone
establishes broad superiority or sufficient ICLR novelty. The contribution is
promising but must be stated at this narrower level.
