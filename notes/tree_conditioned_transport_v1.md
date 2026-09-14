# Retaining latent-tree information inside the flow

The adapter comparison is COMPLETE and does not establish molecular utility. The design
motivation and joint source/target contract are in
`notes/tree_conditioned_transport_brief_v1.md`. Prior results remain unchanged.

All3,072 outputs replay in `research/evidence/tree_context_audit_v1.json`.
Graph passes/512 for no-context / actual-tree / independent-tree are118/102/119
and97/75/97. Actual minus independent differences are-3.32 percentage points
[-6.84,0.20] and-4.30[-7.81,-0.78]. Actual minus no-context is-3.13[-6.64,0.39]
and-4.30[-7.81,-0.78]. These are conditional paired intervals on the fixed eight
compositions. The model restores correctly, its weights and gradients change,
and source coordinates/context trees replay, but actual source-tree information
does not yield a generation gain in this recipe. Do not scale or retune this
static adapter against these outcomes. The broader source-framework comparison
is separately frozen in `notes/tree_source_transfer_v1.md`.

## Learned component

For each sampled source tree, form three symmetric edge features: adjacency,
log(1 + tree-path hops), and the mean of the two log(1 + degree) values. A3→32→d_edge
SiLU network adds a residual to FlowMol's embedded edge features before its
equivariant message passing. The final linear layer starts at zero, preserving
the warm field's predictions at initialization. The original FlowMol source is
not edited, and these latent edges never serve as chemical-bond supervision.

The sampler retains one context tree throughout each generated trajectory. In
the actual-tree arm this is the tree that generated x0. The sham arm draws an
independent tree from the same composition-conditioned law, using a separate
fixed random stream. Both have identical feature distributions, parameter counts,
initialization, source law and learning rates. A frozen no-context shell model
provides the third comparison. The sham control tests whether information about
the source component matters beyond extra parameters and generic tree features.

## Coupling and density semantics

The typed pairing can permute source atoms before applying shared group-Haar
augmentation. An opt-in return value exposes its complete source index map;
the context matrix follows that map in both node axes. The old default pairing
outputs and diagnostics remain unchanged. Context is attached inside the FM
graph's local scope and is explicitly required for tree-conditioned training.
Rotations leave these three features invariant. Checkpoint restoration installs
the declared adapter before strict state loading.

The joint source is p(T|c) q(x0|T,c). Its invariance under joint relabeling and
rotation permits the same orbit-preserving argument, applied to (T,x0). The
coupling preserves source T marginal and target coordinate marginal; conditional
flows therefore have a coherent idealized mixture interpretation. Conditioning
can reduce the optimal squared regression risk, but that standard identity is
not a guarantee about finite training or molecular graph support.

The existing marginal-flow density interface now rejects tree-conditioned models.
Substituting the marginal source q(x0) into a field conditional on T would evaluate
the wrong probability object. No exact conditional/joint final-density evaluator,
importance ESS or Boltzmann-law claim is added in this experiment. The numerical
sampler is still midpoint64 to T1 plus0.025-A centered noise.

## Frozen test

`tree_context_s0_v1.json` and `tree_context_s1_v1.json` use the same two connected-
training permutations and warm checkpoint as the preceding source studies. Each
new arm receives3,000 FM updates. Backbone learning rate is2e-5; the new adapter
uses3e-4 in both actual/sham arms, with the same global gradient clipping. The
source affinities are fixed. This is not simultaneous source and transport tuning.

All three methods receive64 fresh outputs on each of the same eight development
compositions for each continuation:3,072 outputs, zero new oracle calls. Compare
actual minus sham and actual minus no-context for graph/geometry support,
fragmentation, exceptions, diversity and cost; report each continuation separately.
Do not choose the best source separately by seed, change graph perception or use
old evaluation geometries for fit. The primary scientific question is whether
actual tree information gives a repeatable extra generation benefit.

57 targeted tests pass across the adapter, original tree source, pairing,
clamped-density and condition code. These include actual FlowMol zero-initialization
equivalence, nonzero adapter gradients, rotation/permutation equivariance, strict
checkpoint round-trip, explicit context guards, source-permutation bookkeeping,
batched graph attachment and unchanged default pairing streams. Test success
licenses the implementation experiment, not an ICLR claim.

Conditioned flows, graph conditioning and latent-variable models have substantial
prior art. The candidate contribution needs a useful distinction and empirical
value in this bond-free molecular source/transport setting. Additional source-
affinity utility and composition generalization are not established by its existence.
