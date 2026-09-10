# Frozen gradient-noise diagnostic

The ongoing eight-atom fixed-observation log-variance run shows rising mean work
and energy in its partial training record. Complete its bounded run; do not turn
that trend into a terminal sampling result. Before adding another objective,
measure gradient noise at the same original FM initialization and target.

Let Q_theta be the normalized forward path law, R_phi the unnormalized target
path law, and W=log Q_theta-log R_phi. With phi fixed, the pathwise mean-work
forward gradient is g=gradient_theta E_Q[W]. Differentiating the empirical
variance of B independent works while holding observed paths fixed gives

    h = (2/B) sum_i (W_i - Wbar) gradient_theta log Q_theta(path_i).

Since E_Q[gradient log Q]=0, E[h]=2(1-1/B)g under differentiation/integration
regularity. This is the standard score-function/reparameterization relation,
not a new theorem. Backward log-variance and mean-work derivatives target
*different objectives* and are not related by this factor.

The two forward estimators can have very different variance. For a Gaussian
Q=N(a,sigma^2) and target N(1,1), pathwise derivative is Y-1. A fixed-score
variance derivative contains cubic noise divided by sigma. An analytic test
checks the expected scaling and demonstrates high small-noise variance; it
is an estimator test, not molecular performance evidence.

Production diagnostic: row 5846, actual Q=+1 / multiplicity 1, 300 K and .1
restraint, native mean, 16 transitions, noise annealing exponent .5; 16 fresh
batches of 16 paths, seed 9068, original FM checkpoint with temperature-input
neutralization. Query each endpoint once and reuse its exact force/energy pair
for both estimators. No optimizer step, clipping, checkpoint selection, replay
or target modification. Report forward and backward means, trace covariance,
mean-gradient RMS standard error, and forward scaled-mean direction agreement.
Budget: 256 oracle queries. This finite diagnostic can identify large observed
variance, but cannot prove equality or estimate asymptotic SNR precisely.
