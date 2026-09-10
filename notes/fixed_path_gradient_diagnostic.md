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

## Completed diagnostic and decision

The 256-query molecular diagnostic completed without updating weights. Forward
trace gradient covariance is 6.6804e7 for pathwise mean work and 8.8353e9 for
fixed-score log variance. After division by the required 1.875 squared, the
latter is about 37.6 times larger. Sixteen batches give noisy mean directions
(cosine .492); this is a descriptive variance measurement, not proof of an
estimator bias. Backward mean-gradient norms are 2178 and 157762 respectively,
but these differentiate different objectives.

The completed 500-update LV experiment fails: final mean work 5811.40, ESS 1/256,
99.22% overlap heuristic; xTB 25/32 converge, successful-only median strain
316.30 eV versus initial 32/32 and 4.405 eV. Do not scale up this recipe.

Sanokowski et al., Rethinking Losses for Diffusion Bridge Samplers, NeurIPS 2025,
https://arxiv.org/abs/2506.10982, is directly relevant prior work. It distinguishes
on-policy LV and reverse-KL updates when both drift directions are learned and
studies score-function reverse-KL alternatives. The forward-only equality above
does not make the joint LV and KL objectives equivalent. No novelty is claimed
for that distinction or for selecting a reparameterization estimator.

The earlier 500-update mean-work runs still descend over the final 100 updates.
A continuation to 1500 total updates will test training duration before another
objective change. Continue all three earlier fixed-joint / annealed-joint /
annealed-energy arms from step 500, preserving their original optimizer, seed,
target and noise schedule. The fresh run adds 1000*16 + 2*256 = 16,512 oracle
queries; cumulative cost is 25,024 including the original 8,512. A matched HMC
control uses eight starts and 3,127 force updates per chain, also 25,024 queries.
Report continuation execution smokes separately from training costs.
