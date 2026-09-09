# Tail coverage and symmetry averaging

The current target on an orthonormal COM-free space H is
gamma(z)=exp(-(E(z)+kappa ||z||^2/2)/kT), with fixed composition/electronic state.
Removing translation is not confinement; kappa is an explicit part of the
target. The following are standard importance-sampling facts, not new theorems.

## Why positive support was insufficient

Assume E is bounded above and below for this fixed system. The target tails
then have Gaussian quadratic decay with variance s_G^2=kT/kappa. A finite
mixture with bounded centers and common variance sigma^2, even after averaging
rotations and atom permutations, has

    log q(z) = -||z||^2/(2 sigma^2) + O(||z||).

Consequently the quadratic term in log(pi(z)^2/q(z)) is

    (1/(2 sigma^2) - 1/s_G^2) ||z||^2.

If sigma^2 < s_G^2/2, the second moment of the ordinary importance weights is
infinite under those assumptions. In the development target kT=1 eV and
kappa=.1 eV/A^2, s_G^2=10 A^2. Both the .3-Angstrom template kernels and the
unit Gaussian lie below the threshold. A good-looking finite-batch ESS does
not diagnose this tail problem. This argument concerns direct importance
sampling; it is not asserted as a theorem about every finite SMC design.

## Defensive component supplied by the target constraint

Use q_D=(1-epsilon) q_local + epsilon q_G, with
q_G=N(0,s_G^2 I), 0<epsilon<1. If E(z)>=E_min, then

    gamma(z)/q_D(z) <= Z_G exp(-E_min/kT)/epsilon = M,
    Z_G = (2 pi s_G^2)^(dim(H)/2).

Thus the direct unnormalized weights are bounded. The geometric bridge
gamma_b=q_D^(1-b) gamma^b has incremental factors r(z)^(b_new-b_old), r=gamma/q_D.
For a fixed increasing schedule, each normalizer increment is a weighted
average bounded by M^(b_new-b_old). Multiplication bounds the SMC normalizer
estimate by M. This gives finite variance under the ideal sampling identities.
The bound is conditional on a genuine global lower bound for E; an observed
sample minimum is not such a bound. It may be extremely loose and is not a
guarantee of useful finite-budget accuracy.

This is classical defensive importance sampling, adapted to the known
confinement factor, with no claim that we invented defensive mixtures:
[Hesterberg (1995)](https://doi.org/10.1080/00401706.1995.10484303),
[Owen and Zhou (2000)](https://doi.org/10.1080/01621459.2000.10473909).
Related adaptive kernel-mixture sampling is also prior work:
https://arxiv.org/abs/1903.08507 .

## Why symmetry averaging complements the tail component

Let transformations g preserve volume, the domain and pi. For any probability
measure over such transformations, define q_bar(z)=E_g[q(gz)]. Jensen gives

    integral pi^2/q_bar <= E_g[integral pi^2/q(gz)] = integral pi^2/q.

Therefore symmetry averaging cannot increase chi-square divergence, when the
integrals are finite. This also holds for a fixed finite subset of valid
transformations, even if the result is not fully group invariant. The Gaussian
q_G is invariant under these symmetries, so averaging commutes with adding the
defensive component. The defensive floor makes the variance argument finite;
symmetry averaging can then reduce it. These observations are standard Jensen
and change-of-variable arguments, not sufficient research novelty by themselves.

The rotation-integrated component uses deterministic matrix-Fisher quadrature.
Sampling identities above refer to its exact integral. The code checks
successive numerical refinements, retains their diagnostics and refuses unresolved
queries; these checks do not supply certified global error bounds.

## Registered comparison

Three seeds, the same 32 particles and 32 annealing stages, with matched eSEN
calls: confinement Gaussian alone, the light-tailed symmetry proposal, the
defensive ordinary mixture, and the defensive symmetry mixture. epsilon=.2 is
fixed in advance. Track the ancestry of defensive-component particles as well
as total ancestry, endpoint ESS, normalizer estimates and geometric spread.
The eight-atom condition and three-atom AgBr2 condition remain development
cases. An independent rotation-reduced randomized-QMC reference for AgBr2 uses
only analytic Gaussian proposals and eSEN queries, without learned templates.
It reports refinement and independent-scramble uncertainty, rather than claiming
exact thermodynamic truth. The AgBr2 doublet state is now confirmed by exact
raw-source replay (raw row 118392, legacy validation row 1137).

Only after these estimates agree and robust physical benefits appear should
weighted teacher populations train a new conditional FM student. Correcting
the proposal, retaining rotational volume and preserving raw electronic states
are necessary repairs; the final ICLR contribution still needs matched empirical
value beyond the cited methods.
