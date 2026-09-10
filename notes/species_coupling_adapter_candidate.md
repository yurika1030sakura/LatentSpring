# Prospective nonlinear exact-volume species coupling adapter

This is the next architecture design to test. A nonlinear centered primitive
is now implemented and unit-tested; the full molecular coupling model is not
implemented or validated. The exact-entropy linear baseline has a small development KL decrease
but leaves ESS~1/256. More flexible invertible transport can change the generator
without requiring an unqualified marginal score estimator.

## Decomposition and triangular updates

For each atomic-number group g, let n_g be its atom count, mu_g its arithmetic
centroid, and u_i=x_i-mu_g its internal displacements. The COM-free space splits
into within-group subspaces (dimension3 sum_g(n_g-1)) and type-centroid space
(dimension3(G-1)), with sum_g n_g mu_g=0. These are counts, not atomic masses;
they match the project's declared unweighted COM measure.

Internal-group layer: replace active-group coordinates by mu_g when forming the
conditioner context; keep all other coordinates. This context is independent
of u_g. Set u_i'=A(context)u_i for every atom in that group, preserving mu_g.
A=exp(B) for a symmetric equivariant3-by-3 B. The inverse recomputes the same
context and uses exp(-B). The exact COM-free log-volume is (n_g-1) trace(B).
Skip groups with n_g=1. Shared A preserves permutation symmetry within the group.

Centroid-pair layer: for two distinct element groups g,h, fix
m=(n_g mu_g+n_h mu_h)/(n_g+n_h) and active delta=mu_g-mu_h. Form context by
replacing both centroids with m while retaining their internal displacements
and every other atom. This context is unchanged when delta is updated.
Set delta'=A(context)delta+t(context), then
mu_g'=m+n_h delta'/(n_g+n_h), mu_h'=m-n_g delta'/(n_g+n_h).
Translate each group by its centroid change; internal displacements remain fixed.
The inverse uses the unchanged context. Exact log-volume is trace(B), regardless
of group sizes, because the fixed linear coordinate change cancels.

Use a deterministic layer schedule based on atomic-number groups (e.g. ordered
internal updates and adjacent centroid pairs, with forward/reverse sweeps), not
coordinate sorting or arbitrary atom IDs. The context projection must be shown
invariant under each layer; dense Jacobian tests must include every context
dependence rather than detaching the conditioner and checking a partial Jacobian.

## Conditioner and objective

A small invariant message-passing network can predict scalar weights using
atomic numbers, active roles, charge, spin and requested target temperature.
Relative context vectors v_i=x_context_i-m generate an equivariant shift and
symmetric tensor, for example
B=(b/2)tanh(b0)I+(b/2)mean_i[tanh(b_i) v_i v_i^T/(||v_i||^2+rho^2)].
Then ||B||<=b; b=.25 is a proposed initial bound. Use a bounded equivariant
shift, initially zero. Zero final heads make each layer identity. Never query
the physical oracle on the masked/collapsed conditioner context; only score
actual generated outputs. Do not modify the shared FlowMol installation.

Compose layers and sum their exact log-volumes. With the base generator frozen,
train E[U(T_theta x)/kT-logdet_H DT_theta(x)]. Independent paired differences
measure marginal KL change; the base density and target normalizer cancel.
Carry any original path work as W_new=W_old+(U(Tx)-U(x))/kT-logdet. Under the
inverse-transformed original auxiliary its approximation gap is constant.
No auxiliary dimensions or noisy CNF log-determinants are introduced in this
design. This objective and change-of-variables identity are established methods.

## Limits, prior art and required gates

This architecture is not universally expressive. In a single-element system,
masking the only internal group erases geometric context, yielding at most an
isotropic map. Two singleton element groups also give limited context. State
these failures explicitly and retain such cases; do not claim coverage of every
OMol25 system or solve homogeneous LJ benchmarks with this design by assumption.
The base FlowMol still defines the proposal. Nonlinear radial alternatives or
an explicitly different fallback would need their own proofs and experiments.

Closest primary prior art includes SE(3) Equivariant Augmented Coupling Flows
(NeurIPS2023), https://arxiv.org/abs/2308.10364 , and Equivariant Finite Normalizing
Flows, https://arxiv.org/abs/2110.08649 . The general equivariant-coupling argument
is not new. Determine whether this chemical-group decomposition is a useful
architectural distinction before any novelty claim. Compare actual performance
and cost to relevant coupling-flow and sampling baselines.

Before molecular training: test exact inverse, COM, joint coordinate/element
permutation and O(3) equivariance, full intrinsic Jacobian/log-volume, parameter
finite differences, identity initialization, singleton/homogeneous edge cases,
and coefficient bounds. Then run a bounded real-oracle smoke with replay checks,
followed by matched linear/scalar controls and independent geometry/distribution
assessment. No test, allocation or molecular outcome for this nonlinear design
has occurred yet. The ICLR objective remains open.


## Preferred nonlinear primitive now implemented

The common3-by-3 A-only internal layer above is a baseline: it preserves
within-type affine invariants and is too restrictive for large element groups.
The implemented alternative is cfm_mol/centered_convex_flow.py. With context
held fixed, use a pointwise gradient map
F(u)=lambda*u+sum_j a_j sigmoid(b_j dot u+c_j)b_j+a_r*u/sqrt(ell^2+||u||^2).
Normalize ||b_j||<=1/ell, set |a_j|<=2*beta*lambda*ell^2/M and
|a_r|<=beta*lambda*ell/2, with beta=.25. The softplus and radial terms each
consume half the derivative budget, giving ||DF-lambda I||<=beta*lambda.
lambda=exp(.25*tanh(raw_scale)); zero amplitudes and scale give exact identity.
The total potential is strongly convex despite the bounded signed coefficients.

On a group zero-sum subspace, T_i(u)=F(u_i)-mean_j F(u_j). Let A_i=DF(u_i).
The exact volume is
 logdet_H DT=sum_i logdet(A_i)+logdet(mean_i A_i^-1).
This follows from the Schur complement of the3-dimensional translation
subspace. It accounts explicitly for centering; simply multiplying the ambient
block determinants is wrong. Only3-by-3 matrices need factorization.

The inverse iteration u_next=(y-P(F(u)-lambda*u))/lambda is a global contraction
with factor<=beta. The implemented no-grad inverse checks a declared residual
and rejects non-convergence; it is for reconstruction, not a differentiable
likelihood API. No inverse is needed for forward energy-minus-log-volume
training. Five tests cover full intrinsic Jacobian agreement, parameter finite
differences through the map and determinant, symmetry, identity and inverse at
up to200 points. These are engineering/theory checks, not molecular performance.

Next implement the invariant context conditioner and species-group wrappers.
For internal groups, use this centered map. For centroid-pair3-vectors, use the
uncentered pointwise map and its3-by-3 determinant, plus a bounded equivariant
context shift if needed. Test complete context dependencies, global COM and
inverse composition before any molecular training. Do not claim universality:
with little passive context, angular expressivity remains limited even though
the added radial term is nonlinear. Keep those cases in assessment.

Contractive residual flows, convex-potential flows and Schur-complement
identities are prior art. The potential research distinction is the molecular
conditioning/decomposition and tractable constrained volume, which still needs
novelty review and controlled performance against equivariant coupling baselines.


Additional primary prior art for this primitive:
Convex Potential Flows, https://arxiv.org/abs/2012.05942 ;
Residual Flows for Invertible Generative Modeling, https://arxiv.org/abs/1906.02735 .
The contraction proof and Schur-complement identity are established mathematics.
No molecular training of the nonlinear primitive has been performed.
