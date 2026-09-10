# Quantitative Gaussian terminal-noise diagnostic

This result combines standard convexity, Gaussian integration and importance-
sampling second moments. No novelty claim is made. The general connection
between chi-squared divergence and importance-sampling cost is reviewed by
Agapiou et al., Statistical Science 2017, https://arxiv.org/abs/1511.06196.

## Exact Gaussian statement

Let p=N(0,Sigma) in d orthonormal coordinates, with covariance eigenvalues t_j>0.
A sampler whose last independent noise is N(0,s I), s>0, has endpoint density
q=mu*N(0,s I), for some probability measure mu. Then its population importance-
sampling ESS fraction is at most

    rho_* = product_{j:t_j<s} sqrt(t_j*(2*s-t_j))/s.

This is sharp over arbitrary mixing measures: choose
q_*=N(0,diag(max(t_j,s))) in covariance eigen-coordinates, realized by mixing
with covariance diag(max(t_j-s,0)). It need not be attainable by the particular
bounded neural drift parameterization. It is NOT an upper bound on a realized
finite-sample ESS, which can exceed the population limit by sampling fluctuation.

## Proof

Write I(q)=integral p^2/q = 1+chi2(p||q). Direct Gaussian integration gives
I_*=I(q_*)=1/rho_*. For a translated noise component phi_s(x-m), another Gaussian
integral gives

    J(m) = integral p(x)^2 phi_s(x-m)/q_*(x)^2 dx
         = I_* exp[-sum_{j:t_j<s} m_j^2*(s-t_j)/(s*(2*s-t_j))]
         <= I_*.

Directions t_j>=s contribute exactly one to this integral because p_j=q_*j.
Convexity of 1/u implies pointwise 1/q >= 1/q_* -(q-q_*)/q_*^2. Integrating
against p^2 and using Tonelli for the nonnegative component integrals yields
I(q) >= 2 I_* - integral J(m) mu(dm) >= I_*. The stated mixing law attains it.
The result also holds for nonzero Gaussian means by translation and for general
positive covariance by orthogonal rotation, since the terminal noise is isotropic.

For the normalized finite-path importance weight A, conditional expectation at
an endpoint gives E[A|X]=p(X)/q(X), because the auxiliary backward kernels are
normalized. Conditional Jensen therefore gives E[A^2]>=I(q). Auxiliary reverse
kernels cannot evade this Gaussian population-ESS ceiling. For unnormalized
weights, divide A by the true normalizer in this argument.

## Molecular scope

The actual restrained eSEN target is not Gaussian. Substituting t_j=kT/h_j from
positive local Hessian eigenvalues is only a harmonic-surrogate diagnostic.
Our saved four MALA endpoints are not certified minima and have negative
curvatures; discarding those directions does not turn the result into a bound
on the full molecular distribution. Rotational modes and global basin masses
are not covered by this substitution. The calculation must not be used to
certify molecular ESS or to attribute all observed collapse to final noise.


The four saved positive-curvature surrogates give fixed-noise population ceilings
.2901, .3305, .3786 and .2967; annealed-noise ceilings are one in each surrogate.
These ceilings do not predict the actual observed molecular ESS fraction near
.004. They show what the local Gaussian model constrains, while optimization,
non-Gaussian structure and global coverage remain unresolved. Values and full
scope are in research/evidence/local_gaussian_noise_surrogate_v1.json.
