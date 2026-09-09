# Terminal Gaussian noise and molecular curvature

For an endpoint X=M+sigma*epsilon in orthonormal COM-free coordinates, with
epsilon standard Gaussian independent of any mean law M, the standard Gaussian
posterior covariance identity gives

    Hessian log q(x) = -I/sigma^2 + Cov(M | X=x)/sigma^4 >= -I/sigma^2.

This is the matrix form of the established second-order Tweedie identity, not
a new theorem. See Efron's [author slides, page 6](https://efron.ckirby.su.domains/talks/2010TweediesFormula.pdf)
and [2011 paper](https://www.tandfonline.com/doi/abs/10.1198/jasa.2011.tm11181).
The covariance is positive semidefinite. Consequently exact equality with a
smooth Boltzmann target exp(-U/kT)/Z requires Hessian U <= (kT/sigma^2) I locally.
This is a necessary condition, not a sufficient approximation criterion.
Importance weights can still correct a broader proposal in principle.

The current reference kernel has final variance

    sigma_last^2 = (kT/kappa) [1-exp(-nu_last^2*dt)].

For 16 equal steps, nu=.2 and kappa=.1, the corresponding curvature ceiling is
40.0500 eV/A2, independent of kT. Force finite differences on fixed indices
0--3 of the 300-K MALA endpoint panel give maximum eigenvalues about
185.31, 209.36, 208.75 and 233.10 eV/A2 at h=.003 A. Results at h=.01 A differ
by at most .125 eV/A2. The 336 queries and full spectra are retained in
research/evidence/terminal_noise_curvature_probe.json. These are numerical
local probes, not certified global Hessian bounds or equilibrium samples.
They support a terminal-noise resolution problem at molecular configurations.

A prospective square-root noise schedule uses nu(t_left)=.2*sqrt(1-t_left).
At 16 steps it raises the ceiling to 640.0500 eV/A2 and makes the 300-K final
standard deviation about .00635 A rather than .02541 A. All finite-step
variances remain positive, and actual forward/backward Gaussian factors are
evaluated. Zero-residual Gaussian-reference work remains constant; this and
full/checkpointed gradients are tested for both schedules. Falling below the
observed curvature ceiling does not guarantee accurate sampling, global mode
coverage or successful backward-kernel fitting. Smaller noise can make density
ratios harder to fit, so fixed-noise controls remain necessary.

The next 300-K calibration compares, with the same initialization and 8,512
potential queries per arm: fixed-noise joint training on condition5846,
square-root-noise joint training and its energy-gradient control on5846, and
square-root-noise joint training on AgBr2 condition1137, where independent
normalizer/moment estimates are available. Every arm uses500 updates, batch16,
16 transitions, 256 evaluation samples and seed9051. Temperature-feature
initialization is neutralized as described in the mean-work protocol.
This is a test of an existing noise-scheduling remedy and its molecular effect;
neither the schedule nor Tweedie's identity is claimed as new.

Exact-likelihood alternatives are also prior work: [SE(3) Equivariant Augmented
Coupling Flows](https://arxiv.org/abs/2308.10364), NeurIPS2023, already provide
equivariant augmented coupling constructions for molecular distributions.
Any future invertible replacement must compare against that work as well as
RegFlow and FALCON, rather than claiming the coupling construction itself.
