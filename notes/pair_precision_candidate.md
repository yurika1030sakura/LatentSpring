# Candidate: learned pair precision for finite-work molecular transport

Status: implementation candidate, not established AI novelty or molecular benefit.
The user explicitly requested developing a stronger AI contribution. This bounded
prototype tests a concrete geometric change while retaining the original target,
FlowMol initialization and normalized finite-path probabilities.

## Hypothesis and construction

An isotropic noise scale couples resolution of stiff internal motions to noise
in softer collective directions. Test whether a learned, bond-free precision
matrix can improve the calibration/compute tradeoff through selective correlated
perturbations. The current local Gaussian ceiling does NOT establish this as the
cause of molecular collapse, so the claim requires actual controlled experiments.

For each unordered pair define u_ij=(x_i-x_j)/sqrt(||x_i-x_j||^2+.1^2) and its
block displacement direction g_ij. Let 0<=w_ij<=1 be a symmetric function of atom
embeddings, distance, time, total charge, spin and temperature, multiplied by
exp(-r_ij^2/2^2). A small neural network learns these weights. In an orthonormal
COM-free basis T, use the relative precision

    P(x,t) = I + alpha/N * T^T [sum_{i<j} w_ij g_ij g_ij^T] T.

The bracket is an elastic-network-type matrix. Since each pair outer product
is bounded by its scalar complete-graph Laplacian block, I<=P<=(1+alpha)I.
Its geometric term annihilates infinitesimal rigid rotations. Translation,
rotation/reflection and atom permutation transformations preserve the density
law when atom identities are transformed together. These facts do not imply
chemical validity or a molecular equilibrium distribution.

For the existing scalar step variance s^2, the conditional covariance becomes
s^2 P^-1. Cholesky sampling and the complete quadratic/log-determinant Gaussian
factor are used in BOTH directions. This changes the sampler but keeps its
finite-path weights exactly specified. The scalar reference coefficients no
longer define a reversible Gaussian process when P depends on the state; only
the general normalized-kernel identity is invoked. Initial q0 is unchanged.
The new precision head is shared by forward and backward kernels, and all of
its pathwise derivatives are retained under the mean-work objective.

## Controls and limitations

Required arms: learned pair precision; frozen geometric precision with w=.5
times the same envelope; isotropic covariance with the same trace as P^-1 at
the same input; existing isotropic fixed/annealed kernels. This separates learned
weights, geometric preconditioning, anisotropy and overall variance reduction.
A changed initialization distribution is recorded through each arm's initial
evaluation. No improvement is inferred from a changed noise norm alone.

Initial scales: alpha=32, pair softening .1 A, distance envelope 2 A; precision
learning rate .001 versus mean-network rate 1e-5. The dense COM matrix has cubic
factorization cost; scalability to 200 atoms has NOT been demonstrated. Keep the
model's max_atoms=200 and bond-free data convention unchanged.

Execution gate: eight-atom row 5846, original Q=+1/singlet, 300 K, .1 restraint,
native means, constant scalar noise .2, two updates, B2, 64 before/after paths,
seed 9076, 132 oracle queries. A batch-16 gate and frozen-checkpoint reproduction
must pass before bounded training comparisons. Initially only joint mean-work
training supports this head; LV/gradient-diagnostic ablations are rejected.

## Prior art, checked September 10, 2026

- Elastic-network matrices are established; e.g. the original authors' ProDy
  ANM documentation: https://www.bahargroup.org/prody/manual/reference/dynamics/anm.html.
- AniDS already learns structure-aware SO(3)-equivariant anisotropic noise for
  force-field pretraining: https://papers.neurips.cc/paper_files/paper/2025/hash/5a1b803bd55226b097f38354353cc467-Abstract-Conference.html.
- Chroma uses correlated diffusion for proteins:
  https://www.nature.com/articles/s41586-023-06728-8.
- SNF/FEAT and learned diffusion coefficients remain direct prior work.

Thus anisotropy, correlation, a graph metric or a Gaussian work identity cannot
be claimed as new by themselves. A potential contribution would need a distinct
learning mechanism plus predicted and replicated benefits under the above
controls. The present search is not an exhaustive novelty certificate.

## Verification

228 tests pass. Added tests cover spectral bounds, rigid rotational directions,
translation/rotation/reflection/permutation covariance, collision regularity,
trace matching, coordinate gradients, Gaussian density against an independent
API, sampled covariance, scalar-kernel recovery and covariance/mean parameter
gradients with checkpoint recomputation and finite differences.
