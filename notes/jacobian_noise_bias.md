# What the stochastic residual actually regularizes

This is a derivation from standard Gaussian/Rademacher quadratic-form moments,
not a novelty claim. It clarifies how probe coupling changes the energy loss.
All formulas condition on deterministic trajectories and fixed geometries.

Let J_{i,s} be the COM-projected velocity Jacobian for geometry i at quadrature
stage s, h_s its quadrature weight, and S_{i,s}=(J_{i,s}+J_{i,s}^T)/2. Define
U=S for Gaussian probes, or U=offdiag(S) for Rademacher probes. The covariance
of two centered quadratic forms driven by the same probe is

    Cov(xi^T J_i xi, xi^T J_j xi) = 2 <U_i,U_j>_F.

For K geometries, R independent averaged replicas, and probes refreshed at
every time stage, the extra expected squared-residual penalty is:

    Independent geometries:
      B_ind = (2/R) (1-1/K) sum_s h_s^2 mean_i ||U_{i,s}||_F^2.

    Common probes within each parent:
      B_common = (2/R) sum_s h_s^2 mean_i ||U_{i,s}-mean_j U_{j,s}||_F^2.

The proof substitutes the covariance into tr(C Sigma)/K and uses the identity
mean ||U_i||^2 - ||mean U_i||^2 = mean ||U_i-mean U_i||^2. Thus common probes
change a Jacobian-magnitude penalty into a within-neighbourhood Jacobian-
dispersion penalty. The latter vanishes for affine fields with a shared
Jacobian, explaining the affine control; it does not vanish in general.

For a probe fixed over the entire trajectory, replace U_{i,s} by the symmetric
part (or off-diagonal symmetric part) of A_i=sum_s h_s J_{i,s}, and omit the
outer sum over stages. Cross-time covariance then remains. For bounded
Jacobians and a uniform grid, refreshed-probe bias is O(1/S), while a fixed-
probe estimate generally retains O(1) variance. The molecular evaluation
panel and the training code therefore cannot share a numerical noise estimate
without accounting for their different probe policies.

Gaussian B is invariant under simultaneous orthogonal coordinate changes.
Rademacher B generally is not because discarding diagonal entries depends on
the coordinate frame. Averaging rotations does not eliminate the extra
objective term. Removing this implicit regularization also does not guarantee
better optimization: the independent-product estimator has substantial
variance, including negative realizations.

These effects are related to deliberate Jacobian regularization of neural
ODEs, studied by [Finlay et al. (ICML 2020)](https://proceedings.mlr.press/v119/finlay20a.html).
The formulas above describe the particular penalty introduced by the grouped
stochastic energy objective; they are not that paper's regularizer or a new
general regularization principle.
