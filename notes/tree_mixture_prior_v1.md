# Learned spatial priors from latent connection trees

Current hypothesis: learn a bond-label-free spatial starting distribution for
the main flow-matching generator. This acts on the generator's source law, not
another post-generation MCMC scorer. The connection network and prior fitter are
implemented. A molecular advantage and distinct ICLR contribution are unproven.
Read `research/TREE_PRIOR_STATE_20260913.json` and `research/NEXT.md` for execution.

## The observed failure being addressed

For the stronger typed-matching generator,272 of512 outputs are disconnected by
the fixed contact rule; only6 have severe overlaps, and4 have both. Thus272 of
274 geometric failures involve disconnection. Increasing a collision cost does
not directly address this observed dominant failure. The preceding collision-
pairing comparison remains closed; no negative result is overwritten.

The graph readout also has a representation limitation. Its charged-fragment
mode rejects an idealized neutral methyl radical, while the radical mode accepts
it with one radical electron. On old condition5,26 of35 geometrically supported
typed-model outputs admit the alternative radical assignment, but25 have three
or more radical electrons. Those assignments do not establish the requested
quantum state or energy. No original graph counts or failed experiment gates are
retroactively changed. Full attribution is in
`research/evidence/generator_fragment_diagnostic_v1.json`; the API behavior is
documented by [RDKit](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html).

## What the connection network learns

The inputs are atomic species, pooled composition features, atom count, total
charge and spin multiplicity. The network sees no candidate coordinates or
observed chemical bonds when producing prior edge affinities a_ij(c)>0. Coordinates
are used in maximum-likelihood fitting of the resulting spatial distribution.

A latent spanning tree T has probability

    p_theta(T|c) = product_(ij in T) a_ij(c) / tau(a(c)),

where tau sums the edge products over all spanning trees. For each edge, sample
an independent displacement with an isotropic normalized density h_ij in R^3.
Integrate these displacements along the tree, then subtract the unweighted COM.
The tree is an auxiliary dependence structure, not a proposed chemical bond
graph. Tree connectivity does not ensure contact-graph connectivity, valid
valence, absence of nonedge clashes or a thermally correct final molecule.

The edge radius is lognormal with fixed log standard deviation0.2 and mean
equal to the sum of covalent radii; its direction is uniform on the sphere.
Writing length as ell and mu=log(ell)-sigma^2/2, the vector density is

    log h(d) = -(log r-mu)^2/(2 sigma^2) - 3 log r
               - log(4 pi sigma sqrt(2 pi)),     r=||d||.

This includes the radial/solid-angle Jacobian. It differs from a zero-centered
harmonic Gaussian edge law; neither law is an electronic potential-energy model.
The current differentiable implementation rejects exact pair coincidences rather
than silently clipping them. The sampled continuous law hits these with probability
zero; this restriction is relevant when evaluating arbitrary numerical inputs.

## Exact marginal density on the COM subspace

Let X belong to H={sum_i x_i=0}, with intrinsic dimension3(N-1). For any tree,
the absolute Jacobian from orthonormal H coordinates to edge displacements is
N^(3/2). Hence the normalized conditional density is

    q(X|T,c) = N^(3/2) product_(ij in T) h_ij(x_i-x_j).

Summing over the latent tree yields a tractable normalized coordinate density:

    q_theta(X|c) = N^(3/2) tau(a_ij(c) h_ij(x_i-x_j)) / tau(a_ij(c)).

This is the actual marginal over all trees, not the density conditional on a
single sampled tree. The tree-normalizer gradient has an interpretable form:

    d log q(X|c) / d log a_e
      = P(e in T | X,c) - P(e in T | c).

Thus maximum likelihood learns connection preferences from coordinates without
pseudo-bond targets or a stochastic gradient through a discrete tree sample.
Normalization and learning follow established tree-ensemble identities; this
derivation is not claimed as a new matrix-tree theorem.

`log_tree_partition` evaluates the grounded Laplacian determinant through
positive star-mesh elimination in log space. Eliminating vertex k adds
w_ik w_kj / sum_j w_kj to each remaining edge, and the eliminated degree is a
determinant pivot. Log-sum-exp/log-add-exp avoid subtractive cancellation and
underflow when spatial factors differ greatly. Cost is cubic in atom count;
max_atoms remains200. Weighted Wilson sampling supplies the exact undirected
tree law in ideal arithmetic. A high-degree algorithmic root improves runtime
without changing that law. The embedding is centered afterward.

## The controlled learned objects

The fixed reference uses a_ij=u_i u_j, with small propensity0.05 for H and the
common terminal halogens and a simple default-valence-derived propensity for
other elements. This is a physical heuristic, not a universal valence model.
Every tree variant has the same fixed radial component law.

- Fixed tree: no learned affinity parameters.
- Node tree: a small network learns atom propensities from species and composition;
  the log-edge correction is the sum of its two node scores.
- Pair tree: a small network also uses the symmetric sum/product of pair embeddings
  to learn relation-specific affinities. Corrections are bounded by2*tanh.
- Gaussian continuation: the strong previous generator's ordinary centered
  isotropic source, with the same additional FM training data and budget.

This distinguishes fixed structural bias, learned degree preferences and learned
pair relations. A benefit of node learning need not imply a benefit of the larger
pair model. A covariance/scale-matched Gaussian would still be relevant before
attributing a positive result solely to higher-order connectivity, since different
tree laws can change coordinate covariance despite fixed edge distributions.

## Flow matching and density interfaces

Sample the prior independently of the data endpoint and supply those positions
to `clamped_fm_path`. All arms use the same type-preserving assignment/rotation
coupling and shared group-Haar augmentation. The tree marginal is invariant under
proper spatial rotations and identical-label permutations; the same symmetry
argument preserves this source law. It is not a Gaussian-score identity.

`log_density_clamped_flow` now accepts an explicit prior-log-density callback.
Gaussian defaults are preserved. The custom prior is evaluated at the integrated
preimage, followed by the usual divergence integral. Direct/checkpointed autograd
supports it; the Gaussian-specific discrete-adjoint training route explicitly
rejects custom priors. The optional callback passes a linear-flow value/gradient
test. An analytic source density does not make finite ODE integration exact.

The current molecular output remains midpoint64 at T1 plus0.025-A Gaussian noise.
No absolute likelihood, importance ESS or Boltzmann-law claim is assigned to those
outputs. A normalized source is useful for future nonequilibrium constructions,
but does not supply the missing final-map/path density automatically. Existing
BGFM loss hooks remain intact; this generator pilot uses FM only.

## Frozen experiment and validation

`research/evidence/tree_prior_protocol_v1.json` specifies four3000-step FM
continuations from the same typed-matching checkpoint, with identical OMol25
data order. Node/pair priors each fit exact marginal NLL on the first1000 of these
same training examples;128 additional train-split rows are used only for prior
validation. No generated evaluation geometry, reserved outcome, bond label or
fresh physical-oracle label enters fitting. The prior NLL is divided by3(N-1)
to control unequal molecule sizes; this is not a temperature or force objective.

Each final model and the unchanged warm generator produce64 fresh samples on
all eight existing development compositions:2560 outputs in total. Preserve
initial/final disconnection, overlaps, graph acceptance, validator exceptions,
diversity and runtime. No physical query is part of this first test. A clear
new-method signal would justify a separately frozen physical/distribution check.

52 targeted tests pass: exact tree enumeration and gradients, extreme log weights,
radial normalization, the intrinsic Jacobian, weighted-tree sampling frequencies,
coordinate/parameter gradients and symmetry, custom-prior flow density, and
existing FM/density/condition regressions. Prior fitting is complete; held-out
NLL/DOF is1.84687 fixed,1.83615 node and1.82766 pair. This small likelihood gain
is not a molecular-generation benefit. The actual generation comparison is active.

## Closest prior art

[ET-Flow](https://arxiv.org/abs/2410.22388) and
[HarmonicFlow/FlowSite](https://www.mlsb.io/papers_2023/Harmonic_Prior_Self-conditioned_Flow_Matching_for_Multi-Ligand_Docking_and_Binding_Site_Design.pdf)
already use graph-informed harmonic priors for molecular flow matching with a
supplied molecular graph. Structured priors or correlated noise alone are not new.
[Meila and Jaakkola](https://publications.ri.cmu.edu/tractable-bayesian-learning-of-tree-belief-networks)
establish tractable factored priors/ensembles over trees.
[Duan and Dunson](https://www.jmlr.org/papers/v24/22-0252.html)
develop spanning-tree likelihoods and dependence-graph inference, including
Gaussian pair-difference constructions. Do not claim latent trees, exact tree
marginalization or the Gaussian-tree idea as novel.

The candidate contribution is the learned, explicit spatial source distribution
without supplied bonds and its measured usefulness for this molecular generator.
This is a narrow prior-art assessment, not an originality certificate. Its ICLR
value still depends on a useful distinction and successful controlled experiments.
