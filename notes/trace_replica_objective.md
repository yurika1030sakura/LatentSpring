# Squared stochastic density residuals: the objective being optimized

Let a group have K geometries, C=I-11^T/K, and deterministic residual
r(theta)=ell_h(theta)+beta E. Here ell_h uses the same deterministic numerical
trajectory and exact trace quadrature. A stochastic trace gives
r_hat=r+epsilon, with E[epsilon | geometries,theta]=0 and covariance Sigma.
Linearity of fixed-step trace quadrature gives this conditional statement;
it does not remove trajectory discretisation error or describe a joint CTMC.

The plug-in loss obeys

    E[ ||C r_hat||^2 / K ] = ||C r||^2 / K + tr(C Sigma)/K.

The extra term generally depends on theta. Increasing training seeds does
not remove this objective bias. A finite-step estimate of log q being
unbiased in its trace does not make its square unbiased.

For two conditionally independent trace replicas on the SAME deterministic
trajectory, define

    L_cross = (C r_hat_1)^T (C r_hat_2)/K.

Then E[L_cross]=||C r||^2/K. Under domination sufficient to interchange
expectation and differentiation, E[grad L_cross]=grad ||C r||^2/K, provided
the trajectory and prior gradients are retained. Freezing the trajectory
invalidates this last assertion. Shared model dropout or shared trace probes
between the replicas invalidate the independence argument.

For equal compute, compare to two-replica mean plug-in loss, not a one-probe
baseline. Pointwise,

    L_mean - L_cross = ||C(r_hat_1-r_hat_2)||^2/(4K) >= 0.

The cross-product itself can be negative. Positivity clipping changes the
objective. The current training safety cap uses the absolute loss magnitude
and records skips; selective skipping is another source of bias and must be
reported, not hidden behind the unbiasedness statement.

For R>=2 independent replicas, the all-distinct-pairs U statistic equals
the squared replica mean minus the across-replica sample variance (averaged
over centered geometries) divided by R. The numerical panel reports both.

The mean over groups is unweighted by group size to preserve the existing
objective. Missing values are masked jointly across replicas. Energy offsets
are removed in float64 before adding log q; this avoids further cancellation
but cannot recover precision lost in archived float32 energy labels.

These are standard quadratic-form/independent-product identities. Prior
work already studies likelihood-noise problems for Boltzmann reweighting,
including Verlet Flows (arXiv:2405.02805) and Flow Perturbation++
(arXiv:2601.21177). LDR (arXiv:2602.03729) covers the underlying off-policy
regularisation framework. Any contribution here needs empirical and algorithmic
evidence specific to training molecular CNFs, beyond these identities.

Implementation: `cfm_mol/replica_loss.py`, with shared-trajectory replicas in
`cfm_mol/clamped_density.py`. Exact enumeration tests check values, parameter
gradients, negative realizations, group offsets and missing-value handling.
This does not yet establish improved molecular sampling.

The initial molecular quadrature panel fixes each probe over time; training
by default redraws probes at each integration stage. These policies have
different covariance, so panel noise magnitudes must not be substituted for
training noise magnitudes. Step-derived dedicated trace generators in the
matched experiments separate trace draws from FM times/priors. Common probes
across sibling geometries are included as an equal-budget control, alongside
the independent product; they are not shared between replicas.
