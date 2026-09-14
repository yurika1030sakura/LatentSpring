# Current-geometry structured connectivity inside molecular flow

Status: implemented candidate; no improvement result yet. The preceding source
utility pilot is complete and negative. The user's latest priority is a fast,
substantive AI-method test with molecular improvement; this is a different bounded
architecture hypothesis, not a retuned source-utility recipe or revived static
source-tree adapter. Its usefulness and originality remain to be established.

The missing capability being tested is state-dependent global structure inference:
source-only adaptation chooses a distribution once, and the old static adapter
supplied the sampled source tree. Neither supplied a global connectivity marginal
recomputed from the actual geometry at each denoiser call. A complete-graph GNN
can in principle learn global information; we do not claim the original field is
mathematically incapable of learning connectivity or that this cause is proven.

For current positions X, physical lengths ell_ij=r_cov_i+r_cov_j and the fixed
species propensities u_i, define positive symmetric edge weights

    log a_ij = log u_i + log u_j - 2 log(1+||x_i-x_j||^2/ell_ij^2)
               + 2 tanh f_theta(h_i+h_j, h_i*h_j, symmetric_edge_embedding,
                                 log(1+d^2/ell^2), 1/(1+d^2/ell^2), ell).

Node summaries h are learned8-dimensional projections of the current field's
invariant node embeddings, which include its time/electronic conditioning. A
32-hidden scalar head predicts affinity adjustments. All distances use the current
positions passed to denoise_graph. No bond graph, future geometry or sampled source
tree is supplied to the module. Current features can be reused in self-conditioning;
the structure is recomputed on every denoiser call, not after every internal GNN layer.

Relative off-diagonal edge weights have a shared1e-8 numerical floor in tree and
local rules; this changes weights, not the tree family. No diagonal ridge or
spanning-forest substitution is used.

Exact undirected tree edge marginals m_ij follow from the grounded weighted-Laplacian
inverse. They have total undirected mass N-1 and sum to at least1 across every
nontrivial cut because every spanning tree crosses that cut. This established
structural property is why weak links between geometric clusters can retain
attention mass. It does **not** guarantee connected coordinates, bonds, valence,
physical forces, or nonzero use of those features by the learned generator.

A zero-initialized edge residual MLP receives log(1+m_ij), degree-normalized m_ij,
and log(1+(degree_i+degree_j)/2), before the existing FlowMol denoiser. Affinity and
message heads are trained by the same main FM objective. No new reward, energy term
or source law is introduced. Rotation/reflection/translation invariance of scalar
features and permutation equivariance preserve the original vector-field symmetries.

Four controls are newly trained from the same warm checkpoint, under each of two
training/data seeds: no block; learned tree affinities; fixed physical tree
affinities with trained messages; learned independent edge mass normalized to the
same N-1 total. Learned tree and local variants have equal capacity/initialization.
The fixed-affinity arm freezes only the affinity head and its node projection;
its message network is learned. It is not a fully unlearned physical sampler.

Frozen protocols: `research/evidence/dynamic_tree_s{0,1}_v1.json`. Each method gets
3000 matching FM updates (base AdamW2e-5, extra module3e-4, clip1); the fixed shell
source, original typed coupling and64-midpoint sampler remain shared. All candidate/
evaluation compositions, including the new utility-held12, are excluded from these
new FM training rows. The no-block baseline is retrained under these same exclusions
and seeds, rather than reusing an unfairly different training run.

The first4 metadata-hash-ranked references per size bin from the earlier monomer
panel provide12 reused development compositions. All are absent from the known
processed training corpus. Each arm receives64 fresh draws per condition,6144
total across two model seeds. This is a reused development benchmark, not an
untouched confirmatory test. Frozen gate: learned tree beats no-block and local
point estimates in both seeds, pooled paired95 lower bounds positive against both,
no >2pp drop in distinct connectivity per attempt against no-block. Report fixed-
affinity performance separately; do not claim learned-affinity benefit without it.
No per-seed winner selection or optimizer/width/rate sweep on these outcomes.

Matrix-tree structured attention predates this work: Liu and Lapata (TACL2018),
https://aclanthology.org/Q18-1005/; Kim et al. (ICLR2017),
https://arxiv.org/abs/1702.00887. General molecular attention also has extensive
prior art. The candidate is the particular state-dependent global connectivity
inductive bias for this bond-free molecular flow task, contingent on useful
controlled results. It is not a new theorem, new physics law, or automatically an
ICLR contribution. The source-utility output-KL bound does not apply when these
newly trained decoder weights change. Final output density/ESS/Boltzmann claims
remain unqualified.
