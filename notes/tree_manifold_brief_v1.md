# Next prototype: learn coordination, transport constrained tree coordinates

Status: the full pilot is complete and its chemical-validity gate fails. Six
implementation tests pass; see `notes/tree_manifold_results_20260914.md`. The completed dynamic-attention pilot also fails to
establish a repeated useful neural gain. Attention supplies information, but does
not preserve any actual coordinate constraint. This next hypothesis changes the
state representation so the sampled spatial tree remains geometrically connected.

## A more expressive normalized tree law

Let k_i=degree_T(i)-1 and fixed coordination caps c_i limit k_i to0..c_i-1.
A composition/electronic-state conditioned neural network predicts nonlinear
potentials psi_i(k). Define p(T|c) proportional to exp(sum_i psi_i(k_i)).
Classical Pruefer enumeration gives

 Z_tree = (N-2)! [z^(N-2)] product_i sum_(k=0)^(c_i-1) exp(psi_i(k))*z^k/k!.

Positive coefficient dynamic programming evaluates the partition and its
parameter gradients. Backward sampling draws degree counts; a uniform permutation
of their Pruefer multiset gives an exactly sampled tree. The probability depends
on degrees, not an arbitrary code order, preserving same-species permutation
symmetry. Terminal H/halogens have cap1; conservative organic N/O/B ceilings allow
common charged local centers inside neutral molecules. These caps constrain the
auxiliary tree only, not every inferred chemical contact or bond order.

This family adds a capability missing from the earlier species-pair source:
with two carbons and two hydrogen leaves, a tree with both H attached to one C and
a tree with one H per C have identical edge-type counts. Species-pair affinity
weights cannot change their relative probabilities. Nonlinear degree potentials
can distinguish the carbon degree profiles(3,1) and(2,2). This is an expressivity
comparison with our old source, not a claim that degree-weighted trees are new.

The standalone prototype is `cfm_mol/degree_tree.py`. Tree partitions/probabilities,
finite gradients and relabeling are checked against exhaustive small-tree sums.
A target spatial tree may be extracted from reference distances before fitting;
no supplied chemical bond labels or bond-order loss is involved. Such geometric
scaffolds are derived structural supervision and must be disclosed as such.

## Flow on a geometry-preserving product space

For each oriented tree edge e, use a real radial logit y_e and unit direction
u_e on S^2. Let r_e=ell_e*(a+(b-a)*sigmoid(y_e)), where ell is the sum of covalent
radii and (a,b)=(0.65,1.20). The tree incidence matrix B has a unique zero-COM
inverse B^T(BB^T)^(-1); reconstruct X from r_e*u_e. Every tree edge stays strictly
inside the declared geometric contact cutoff1.25*ell, including after finite
integration, so raw output coordinates are connected. Non-edge clashes, chemical
validity, physical stability and Boltzmann statistics do not follow.

Training paths interpolate radial logits and use short great-circle paths for
directions. The existing equivariant node-vector network parameterizes radial
logit and tangential direction velocities through oriented edge differences.
The teacher angular speed is at most pi; clipping predicted angular speed at pi
retains the ideal conditional-mean field. Product-space midpoint integrates with
spherical exponential updates and parallel transport, keeping unit directions
and radial support without extra neural function evaluations. This is an
application of Riemannian flow matching, not a new RFM theorem.

`cfm_mol/tree_manifold.py` contains the coordinate map, path, source sampler,
parallel transport and midpoint routine. Roundtrip/finite-difference tests verify
the path velocity; an adversarial large finite vector field still preserves edge
bounds and unit directions. The matched neural runner is implemented in `scripts/research/run_tree_manifold.py`;
its frozen protocols are `research/evidence/tree_manifold_s{0,1}_v1.json`. The marginal coordinate density over degree-constrained trees is
NOT supplied by the old matrix-tree formula. Only the tree law and conditional
initial geometry law are explicit here; no final-density or thermal-law claim.

## Required first comparison, before any performance claim

Use a new, prospectively selected real TRAIN subset within the declared organic
monomer scope, excluding all evaluation compositions. Its reference geometry must
admit the chosen bounded-edge, capped-degree scaffold. Fit the same degree prior
and use the same tree-conditioned network/source for every arm. Compare ordinary
Cartesian FM, its deterministic final tree-edge projection, and a model trained
with product-space RFM. The projection control is essential: connectivity from a
physical coordinate clamp alone does not establish learned-method value. Match
training budget, coupling, source draws and neural evaluations. Use rotation-only
alignment with shared Haar augmentation; arbitrary atom permutation breaks the
fixed-tree condition. Freeze an actual protocol and inspect data eligibility before
training. No new model or result is claimed by this brief.

Primary references checked:
- Chen and Lipman, *Flow Matching on General Geometries*, ICLR2024:
  https://arxiv.org/abs/2302.03660
- Cameron, classical Pruefer degree enumeration notes:
  https://webspace.maths.qmul.ac.uk/p.j.cameron/C50/no7.pdf
- STGG and known-graph molecular internal-coordinate generators remain relevant
  prior art; a dedicated task-specific comparison is still required.


The first pilot uses `CoordinationTreePrior`: a degree-sensitive heavy-atom core
followed by capacity-preserving allocations of terminal species in fixed atomic-
number order. Each allocation has a coefficient-DP normalizer and then uniformly
assigns identical leaf labels, preserving permutation symmetry. The available
capacity is sum(core caps)-2*(N_core-1), independent of core-tree shape; once total
terminal count is feasible, allocation cannot dead-end. Core-degree potentials
can distinguish linear and branched alkane scaffolds even when total carbon degree
is fixed by saturation. Leaf count potentials can learn attachment preferences.
The complete source tree law is explicit; a marginal coordinate density over all
these constrained trees is not claimed. The simpler full-tree degree model remains
a tested mathematical building block, not the final pilot source family.

All three arms share this learned prior and the same tree-conditioned FlowMol
network: Cartesian FM, its final radial-projection control, and product-space FM.
All use128 neural evaluations and no added terminal Cartesian noise in this new
study. The projection reuses Cartesian neural draws. Training uses the same edge-
vector metric, different declared paths and correct velocity interpretation for
each method. Six tests include actual vector-field gradients. The12 reused
validation compositions all have feasible coordination budgets; their coordinates
are never inputs. Two3000-step models per seed are planned, with1000 prior updates
and a matched train-only scaffold-qualified subset. These are development tests,
not an untouched evaluation, a proved AI novelty, or an energy-law result.

The coordination family supplies a capability absent from species-pair affinities;
it is not a strict superset. In particular its heavy-core law has no learned
pair-type affinity. The completed chemical-validity failure must remain distinct
from its correct geometric-connectivity guarantee.
