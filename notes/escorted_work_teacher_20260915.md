# Force-escorted work teacher: exact local identity, empirical distillation

The user requested that the next physics-learning test include Jarzynski. Before
teacher generation or student fitting, the unrun MALA draft was replaced with a
force-escorted instantaneous-quench experiment. Existing harmonic checkpoints and
all evaluated coordinates remain frozen and outside fitting. Training teachers
are newly generated FIT samples from8 metadata-selected TRAIN compositions per
continuation, absent from the10 evaluated compositions.

For an anchor A with accepted perceived graph G, choose a centered Gaussian
reference q0(X|A)=N_H(A,sigma_A^2 I). Its width and a deterministic escort are fixed
before drawing particles:

- sigma_A = min(0.03 Angstrom, sqrt(0.10 Angstrom * kT / ||F_plus(A)||)).
- delta_A = sigma_A^2 F_plus(A)/kT after removing net force.
- X ~ q0; Y=X+delta_A; the intrinsic map determinant is1.
- kT=0.025851999786435 eV, corresponding to300 K.

The unnormalized LOCAL target is
rho_A(Y)=q0(Y|A) exp[-(E_plus(Y)-E_plus(A))/kT] 1_{same graph G}(Y).
It is Gaussian restrained around A. It is not the unrestricted molecular Gibbs
law. E_plus is the inversion-symmetrized conservative eSEN surrogate, and forces
use F_plus(X)=(F(X)-F(-X))/2.

Set H0=-kT log q0 and H1=H0+E_plus-E_plus(A)+hard support. Generalized work is
W=E_plus(Y)-E_plus(A)+kT[log q0(X|A)-log q0(Y|A)], or infinity outside support.
The corresponding exact identity is
E_{q0}[exp(-W/kT)]=integral rho_A(Y)dY.
Both Gaussian reference terms are required: using energy difference alone would
incorrectly ignore the escort's changed proposal distribution. This is an
application of escorted Jarzynski/change-of-variables identities, not a new law.
See [escorted free-energy simulations](https://arxiv.org/abs/0804.3055).

Two tests pass: a constant-force escort has constant complete work (but variable
energy-only weights), and Monte Carlo normalizer/weighted mean agree with an
analytic Gaussian target. Invalid support has zero weight. These mathematical
tests do not validate a global molecular Boltzmann generator.

Eight particles per anchor produce self-normalized weights. This finite-particle
teacher is biased relative to the exact normalized local target; recorded local
particle ESS quantifies concentration only and is NOT generator-output ESS.
All-zero support anchors are retained in records and dropped identically from
all student pools. No replenishment or evaluation-based anchor selection occurs.

Four methods are frozen before training: original harmonic model, replay-target
continuation, unweighted valid escorted-particle continuation, and full-work
weighted continuation. Each student alternates500 original-reference FM updates
with500 generated-FIT teacher updates. Backbone, initialization, selected anchors,
source and randomness agree across students. The work and unweighted escort arms
differ only in particle-selection probabilities. No extra bond loss or invalid
Gaussian velocity-to-score proxy is used. The legacy generator temperature input
stays1; no variable-temperature model is being trained.

Compare graph validity, perceived-connectivity diversity, energy per atom, forces
and all costs. The primary work-versus-escort energy statistic conditions on
paired common-graph support and is reported with its selection boundary. Neither
a positive finite-particle training result nor this local identity establishes
correct chemical-isomer mixture weights, global thermal sampling or ICLR readiness.

Related-method boundary: [Energy-Weighted Flow Matching for Boltzmann sampling](https://arxiv.org/abs/2509.03726) already formulates weighted FM using proposal densities; [energy-guided flow matching](https://arxiv.org/abs/2503.04975) already studies energy-tilted data distributions; [Flow Annealed Importance Sampling Bootstrap](https://arxiv.org/abs/2208.01893) learns flows from weighted physical samples. Local complete work here is mathematically an importance ratio for an anchored target. It cannot be claimed as a new generic importance-weighted FM principle. Any contribution must be the specific molecular construction and demonstrated utility/cost in the declared task.
