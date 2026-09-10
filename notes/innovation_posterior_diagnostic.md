# Frozen-path global innovation posterior diagnostic

This is an alternative normalized auxiliary representation for an unchanged
forward sampler. It is a mechanism diagnostic, not claimed AI novelty.
Gaussian conditioning, Gauss-Newton inference and low-rank determinant/solve
identities are established. See Bell and Cathey1993, IEEE TAC,
https://ieeexplore.ieee.org/document/250476/ . Gaussian flow perturbations and
learned reverse-noise correction are direct prior work (Peng/Gao2025):
https://www.nature.com/articles/s41467-025-62039-8 and the accessible preprint
https://arxiv.org/html/2407.10666v2 . We retain the original16-step forward
sampler rather than replace it with a noisy full ODE map.

## Probability accounting

The original scalar-noise path has K transitions in dimension d. Collect the
standardized initial position and first K-1 Gaussian innovations as z in R^(Kd).
The last endpoint is y=F(z)+s*epsilon with z,epsilon independent standard normal
and s the actual final noise scale. The deterministic map F runs the original
means and internal innovations; it returns the final conditional mean.

The normalized joint density is Q(z,y)=phi(z)N(y;F(z),s^2 I). The transformation
from z to intermediate states x0,...,x(K-1) is triangular with constant logdet
C=d[log(prior_std)+sum(k<K)log(s_k)], where the sum excludes the last transition.
Consequently log Q(z,y)=log Q_states+C and log L(z|y)=log L_states+C for the
existing sequential backward representation. Both ratios must agree numerically.
Neither global invertibility of F nor an endpoint CNF likelihood is needed.

For any normalized auxiliary L(z|y), the work is
W=U(y)/kT+log Q(z,y)-log L(z|y), with the same recorded energy offset as before.
All tested auxiliaries share exact endpoints, energies, forward model and target.
The resulting weights are path weights, not endpoint likelihood values.

## Global Gaussian approximation

Approximately minimize .5||z||^2+.5||y-F(z)||^2/s^2 by4 Gauss-Newton iterations,
with8 per-row backtracking attempts. Initialize only the first latent block as
y/prior_std, leaving other innovations zero. This is an explicit function of
y; it avoids coincident-atom initialization and never uses the generating z.
Both mean and metric depend solely on the endpoint. Saving the generating noise
for later likelihood scoring must not leak it into auxiliary construction.

At the final mean mu(y), compute J=dF/dz and use
L_gamma=N(mu,[I+gamma J^T J/s^2]^-1), gamma in{0,.01,.1,1}, all prespecified.
The normalization and Gauss-Newton solve use d-by-d matrices, not (Kd)^2 ones.
A dense d-by-Kd Jacobian is still expensive; no production scalability is claimed.
Non-convergence and local-Gaussian error remain explicit. Positive covariance
alone does not guarantee finite importance variance. Broadening the metric
changes only the auxiliary, never the requested target or generated coordinates.

## Verification and execution gate

Analytic linear-Gaussian tests use more latent than endpoint dimensions and
compare the auxiliary to the direct multivariate-normal conditional. With the
exact conditional and target equal to the known endpoint marginal, every path
weight is exactly1. A wrong auxiliary restores noise on the same endpoints.
Nonlinear path tests compare states, triangular-density change and directional
derivatives for native/reference means and both noise schedules.

The real model first runs4 particles, batch2, one iteration. It must preserve
states within1e-4 A and log joint densities within .05 nat, with an independent
directional finite-difference ladder (h=.003,.001) agreeing within5% on at least
one resolution. Save both errors, all attempts and frozen-model verification.
The4 oracle calls are engineering cost. If the smoke passes, run32 particles,
batch8,4 iterations, seed9091, at most0.5 GPU-hours and32 oracle queries.
This small result can only justify further independent evaluation if at least
one global arm has ESS>=4/32 and exceeds the sequential auxiliary. Otherwise
stop this global-Gaussian recipe; do not infer that every global conditional
family or forward-learning intervention has been ruled out. No heldout/evaluation
choice is based on molecular energies, which are queried only after inference.
