# Prospective nonlinear exact-volume species coupling adapter

This is the next architecture design to test, not an implemented or validated
method. The exact-entropy linear baseline has a small development KL decrease
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
