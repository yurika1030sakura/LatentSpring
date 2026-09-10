# External coupling-flow comparison contract

Primary source rechecked: https://arxiv.org/html/2308.10364v6 , sections3.3 and
B.7--B.10. Official code: https://github.com/lollcat/se3-augmented-coupling-flows .
EACF splits augmented coordinates and uses a joint density. Its joint reverse-KL
objective upper-bounds the physical marginal KL. The paper uses FAB for its
energy-training experiments and discusses mode-seeking under reverse KL.
Its joint ESS is not an exact marginal-ESS measurement.

Implication for our proposed comparison (standard KL chain rule, not novelty):
if an augmented adapter starts from q0(x)r(a|x) and its target is pi(x)r(a|x),
the initial conditional auxiliary gap is zero. A joint change DeltaJ therefore
satisfies DeltaJ = DeltaM + Gap_final, with Gap_final >= 0. A negative DeltaJ
certifies a marginal decrease of at least that magnitude, but a less-negative
joint change cannot by itself prove that our exact marginal change is better.
Do not rank unlike KL quantities as though both were exact marginal errors.

For an external baseline, preserve source, target, charge/spin, geometry measure
and physical-query budgets, and report auxiliary dimensions, objective,
representation and complete computation. Compare physical-coordinate quality,
diversity and reliable independent distribution diagnostics. Any marginal-density
estimate must include its own Monte Carlo/numerical qualification. A bare lower
joint ESS must not be called worse endpoint coverage. Using Hutchinson CNF
likelihoods for ESS also requires estimator and solver validation.

Our current fixed-base refinement has no auxiliary variables in its adapter and
its change-of-variables objective is an exact relative marginal-KL expectation
under its stated integrability assumptions. This removes that particular
objective gap, but does not establish adequate coverage, a better model family
or novelty. The eight-condition source itself has no evaluated absolute density.
Strong external flow/sampling comparison remains unimplemented.

The possible architectural distinction to test is the element-group decomposition
and nonlinear centered point map with exact3-by-3 volume accounting. General
equivariant coupling, entropy minimization and KL-chain-rule arguments are prior
work. The current small two-seed result on one condition is insufficient for an
ICLR contribution claim; homogeneous-context expressivity limits also remain.
