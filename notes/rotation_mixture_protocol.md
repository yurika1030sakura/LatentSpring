# Molecular symmetry in the explicit proposal

The first four-arm SMC pilot improves endpoint weight ESS to 19.8--28.6/32,
but retains only 2--9 original ancestors. Log-normalizer estimates differ by
up to 16.45 nats between initial proposals, so this does not establish global
sampling correctness or molecular advantage. Arbitrary template orientations
are one possible source of missing coverage; they are not established as the
only explanation. The target and energy scale remain those of the earlier
restrained development experiment.

For intrinsic coordinates Z in R^((N-1) x 3), define the explicit proposal

    q(Z) = (1/M) sum_j integral_SO(3) Normal(Z; C_j R, sigma^2 I) dHaar(R).

The prior is normalized because it averages normalized Gaussian densities.
The Haar measure has total mass one. It is not an optimally aligned Gaussian
treated as though alignment preserved density. Its component log density is

    -(||Z||^2+||C||^2)/(2 sigma^2)
    -3(N-1) log(sigma sqrt(2 pi)) + log c(C^T Z / sigma^2),

where c is the matrix-Fisher normalizer. The score is
(C E[R|Z,C] - Z)/sigma^2. Mixture responsibilities give the full mixture score.
Both c and E[R] follow from standard directional statistics, not a new theorem.

The implementation uses the corrected one-dimensional integral and first
moments from [Lee, Theorem II.1](https://arxiv.org/abs/1710.03746).
It uses scaled Bessel functions and refines Gauss--Legendre quadrature until
successive log-normalizer and first-moment estimates agree within 1e-10.
It aborts if the 1,024-point limit does not pass. These are numerical convergence
checks, not certified error bounds or an algebraically exact likelihood.
The near-rank-one limit uses sinh(k)/k with an explicit small-matrix perturbation
bound. Analytic first moments avoid differentiating singular vectors.

The related molecular denoising connection is already studied by
[Daigavane et al.](https://arxiv.org/abs/2510.03335), which expresses the optimal
rotation-augmented denoiser using a matrix-Fisher distribution. The matrix-Fisher
connection, rotational augmentation and first moments are not our novelty.
The empirical question is whether this normalized proposal is a useful,
computationally feasible correction/teacher for the current molecular FM.

The symmetry arm additionally enumerates permutations among identical elements
when their group has at most 64 members and averages reflected templates.
For larger groups, a fixed seeded subset is used and the report explicitly
denies full permutation invariance. Either finite mixture stays normalized.
Permutations act through the orthonormal COM basis, and do not change the
global charge or declared spin. The target need not share a symmetry for the
proposal to remain a valid proposal; its density must match its actual sampling.

Nine new checks cover the isotropic matrix-Fisher closed form, independent Haar
Monte Carlo, finite-difference moments, diatomic radial normalization, score
equivariance, sample moments, permutation/reflection invariance, the limited
permutation case and rejection of unresolved quadrature. The full suite has
149 passing tests. Both symmetry modes complete a four-particle real eSEN
preflight, with density refinement changes below 1e-10. Their surviving ancestry
is still only one: this tiny run is an interface test.

The registered next comparison uses three seeds (9031, 9032, 9033), 32 particles,
32 fixed annealing stages and the same oracle budget per arm: Gaussian-prior
MALA, ordinary FM-mixture MALA, rotation-marginalized MALA, and the additional
permutation/reflection mixture. Centers are generated independently of each
production RNG. Preserve all seeds and failures. Report normalizer disagreement,
ancestry and geometric spread alongside endpoint ESS, and count the extra
proposal computation. These are development conditions, not a blind test.
