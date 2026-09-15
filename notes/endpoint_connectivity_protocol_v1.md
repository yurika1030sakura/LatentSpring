# Raw endpoint connectivity intervention

One bounded engineering intervention within LatentSpring, not a new core novelty
claim. Prior fixed-tree coordinates eliminated fragmentation while harming graph
support; learned tree attention and relation codes did not repeat useful gains.
This test leaves the learned Cartesian flow and inference unchanged, and applies
coordinate-only support supervision to its predicted endpoint during fitting.

For predicted endpoint Y=X_t+(1-t)v, let d_ij=||Y_i-Y_j||/(r_i+r_j).
The tree term is the mean of max(d_ij-1.15,0)^2 over a minimum spanning tree of d.
The local control replaces it by the mean nearest-neighbor hinge, which can miss
separate internally connected components. Both add the same all-pair overlap term,
sum max(0.65-d_ij,0)^2/N. Weight is fixed at1 times t^2 for t>=0.5, zero earlier.
The auxiliary term acts only on the second-pass endpoint; original two-pass FM
supervision remains unchanged. No bond labels or fixed source tree are targets.
At generic coordinates the tree loss is a minimum of smooth branches; branch
selection is detached. Ties can have nonsmooth, index-dependent subgradients.
No universal output connectivity/chemical-validity guarantee follows from fitting.

Three1000-update continuations per existing harmonic model: replay, local, tree.
Same initialization, FIT rows, RNG streams, optimizer, clipping and learning rate.
Two additional coefficient1 parameter differences are prespecified, each subtracting
the matched replay update from its regularized update and adding it to the frozen
model; only parameters change. They introduce no inference corrector. Parameter
arithmetic and MST objectives are established tools, not originality claims.

Evaluate all six variants (including frozen) on the reused12-composition development
panel,32 draws each, two continuations. Primary: tree_delta beats frozen AND
local_delta in each continuation, with pooled conditional paired95 lower bounds
positive and distinct-connectivity loss<=2pp versus frozen. Merely eliminating
fragmentation fails the gate. No tuning margins/weights/coefficient on outcomes.
The fresh24 and all other evaluation data remain outside fitting; fresh24 is now
observed and cannot be relabeled untouched after subsequent method changes.

Three targeted tests cover global versus local connectivity, finite-difference
gradients, equivariance, a descent step and overlap detection. All pass. Existing
FM/source/generation code is reused. Maximum submitted GPU time is2hours total;
no molecular physical calls are required. Scientific submission readiness remains
false. The independent-generator comparison is still separate unfinished work.
