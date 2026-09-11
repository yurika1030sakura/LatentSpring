# Global pair-potential refinement prototype

Motivation: whole-element contexts lose directional information on a homogeneous
system. Fixed index halves regain context but break permutation equivariance and
freeze each half-centroid. This prototype instead updates all pairs simultaneously.
It is a new candidate in this repository, not an established novelty claim.
Convex-potential and contractive flows are prior art, including
[CP-Flow](https://arxiv.org/abs/2012.05942).

## Map and coefficient contract

For fixed composition/electronic condition and layer, let

    Phi(x) = lambda/2 sum_i ||x_i||^2
             + sum_{i<j,k} c_ijk sqrt(ell_k^2 + ||x_i-x_j||^2),
    T = grad Phi.

A neural conditioner sees element embeddings, charge, spin, temperature and layer
embedding. It is symmetric in i,j and **does not see coordinates**. The radial
lengths are positive. With beta=1/4,

    lambda = exp(.25 tanh(raw_scale)),
    c_ijk = beta lambda ell_k tanh(raw_ijk)/(N K).

All pairs are included. Coefficients are allowed either sign. Sequentially
applying index-ordered pair updates would not have this permutation contract.
Adding coordinate-dependent conditioning without its derivatives would invalidate
the Hessian below; do not make that substitution silently.

For d=x_i-x_j and s=sqrt(ell^2+||d||^2), the pair Hessian is

    H(d) = I/s - d d^T/s^3,       0 <= H(d) <= I/ell.

For any displacement u,

    |u^T (DT-lambda I) u|
      <= beta lambda/N sum_{i<j} ||u_i-u_j||^2
       = beta lambda sum_i ||u_i-mean(u)||^2
      <= beta lambda ||u||^2.

Thus DT has eigenvalues in [(1-beta)lambda,(1+beta)lambda]. Phi is strongly
convex; T is a smooth global bijection. Pair forces sum to zero, so mean(Tx) is
lambda mean(x); restriction to the unweighted COM-free subspace is a bijection.
The inverse iteration x_next=(y-[T(x)-lambda*x])/lambda contracts by at most beta.
Only no-grad reconstruction is implemented, not inverse likelihood differentiation.

## Exact volume and cost

The full Hessian consists of3-by-3 pair blocks: add H_ij to both node diagonal
blocks and subtract it from the two off-diagonal blocks, then add lambda I.
Its three translation eigenvectors have eigenvalue lambda. Therefore

    logdet_H DT = logdet_R^(3N) DT - 3 log(lambda).

The implementation computes the full3N Cholesky and subtracts the translation
term. It does NOT multiply overlapping pair determinants or reuse the old
independent-group3-by-3 centering formula. Complexity is quadratic storage and
cubic dense factorization per layer. This cost must be measured; no scalability
advantage or universal expressivity claim is made.

The paired affine control uses each pair map's exact tangent at zero, replacing
d/s by d/ell. Its neural parameters, conditioning, layers and spectral bound are
identical. On homogeneous systems it reduces to an isotropic linear map; the
nonlinear candidate can change angular/collective structure through pair distances.
A finite radial basis and finite layer count still limit the family.

The frozen-source energy-minus-volume objective remains valid under its existing
integrability assumptions. The Gaussian particle benchmark supplies a known
source density for diagnostic weights; it is not the pretrained molecular source.
Exact ratios do not remove Monte Carlo uncertainty or prove missing-mode coverage.

## Qualification and bounded pilot

Code: cfm_mol/equivariant_pair_adapter.py. Tests check the full intrinsic Jacobian,
parameter finite differences, same-element and joint element permutations,
O(3), batch independence, identity, inverse at200 atoms, and the exact Hessian
extremes at coincident points. These are correctness checks, not performance.

Freeze the DW4 pilot at tau1, Gaussian scale1.6, four layers,1000 updates,
batch128, learning rate1e-3, no annealing and8192 evaluation parents. Families:
pair, pair_affine, whole_convex, convex (the last is a labelled split diagnostic).
Seeds0 and1; all families use identical parent streams within each seed. Each
arm counts160768 analytic target evaluations; eight arms total1286144. No
molecular oracle query is used. Do not restart a failed arm under the same name
or select only positive seeds. A toy PASS does not justify a large LJ grid;
the singular LJ population-integrability issue remains separate.

Compare paired energy/entropy changes, finite-sample ESS, weight concentration,
learned checkpoint replay, elapsed time and failure denominators. A promising
pilot must then face a qualified independent reference and the molecular/strong
baseline tests. It does not itself qualify an ICLR contribution.

## Completed first pilot and controlled capacity follow-up

All eight arms of job45952003 completed and all8192 saved parents per arm replay.
Nonlinear pair-minus-affine changes are -11.045+/-0.170 and -11.390+/-0.234 nat.
Against whole-element nonlinear they are -4.038+/-0.081 and -4.447+/-0.097;
against the non-equivariant index-split diagnostic the pair candidate is worse
by+1.634+/-0.173 and+0.953+/-0.212. Errors are row SEM. Pair endpoint ESS is only
0.116% /0.268%, so the initial prototype does NOT resolve sampling efficiency.
All results, including this negative evidence, are in pair_transport_pilot_v1.json.

Both seeds reach mean absolute coefficient fractions above0.9998 of the spectral
bound in every layer. This motivates one predefined capacity change: beta=.75,
with the same four layers,1000 updates, batch128, two seeds, source and evaluation
panel. The same-neural-parameter affine control is repeated. To separate range
from an immediate change in gradient scale, the parameterization is

    c = beta lambda ell tanh(raw * .25/beta)/(N K).

It exactly preserves the original beta=.25 model and the initial parameter
Jacobian for all tested beta values. The global proof holds for0<beta<1; inverse
iteration allowance is increased to256, with explicit residual failure. Tests
verify initial parameter-gradient agreement and a near-worst-case inverse at.75.
The follow-up is four arms and643072 analytic target evaluations, no molecular
oracle calls. A larger bound is not itself a novelty claim or guaranteed benefit.
