# Next architecture hypothesis: retain the sampled tree in the transport

Status: design brief only. No new adapter, training result or frozen experiment is
claimed. The completed moment controls are in `research/evidence/tree_moment_audit_v1.json`.

## What the completed study licenses

On fresh draws from the same two connected-data continuations, the shell-tree
source beats both unit Gaussian and covariance-matched Gaussian graph acceptance
in both continuations, with positive conditional paired intervals. It does not
consistently beat the harmonic-tree mixture. Thus a useful source structure beyond
a single covariance is supported on this fixed panel, while superiority of narrow
finite-length edges is not. The earlier weak confirmation stream remains evidence
of sampling variability and is not replaced by the new draws. Extra affinity-
network utility and new-composition generalization are still open.

## Concrete implementation gap

`scripts/research/tree_prior_fm.py` samples `(x0, tree)` at training line146 and
supplies only positions to the flow. Generation saves auxiliary trees for audit,
but `sample_clamped_flow` likewise receives no tree conditioning. Consequently
the backbone must infer source dependencies from intermediate coordinates. This
is a valid marginal FM construction, not a correctness bug. It may nevertheless
make conditional displacement regression harder by hiding the mixture component.

The proposed next test is a small neural adapter that supplies the sampled tree's
edge indicator, tree-path distance and degree features to the existing equivariant
message-passing network. It changes what the transport network learns from, while
retaining the same source law. No actual chemical-bond label or new physical law
is introduced. Conditional/latent-variable flow matching and graph conditioning
are established ideas; this application still needs a prior-art and utility case.

## Probability and symmetry contracts

The source is a joint law p(T|c) q(X0|T,c). For each T, a conditional field can
transport its source conditional to the endpoint conditional induced by the
training coupling. If source and target T marginals agree and endpoint X1 retains
the data marginal, their mixture has the intended data marginal in the ideal
conditional-flow limit. This is a proposed model contract, not a proof of utility
or of exact finite numerical integration.

For squared regression, conditioning on additional T cannot increase the Bayes
risk. The difference is E||E[Y|Xt,T,c]-E[Y|Xt,c]||² for the training joint law.
This standard conditional-expectation identity only motivates the mechanism;
finite network capacity and optimization can still fail to exploit the signal.

The source tree MUST follow every source atom permutation during coupling.
`typed_orbit_pair` presently forms source rows with
`permutation[random_permutation]` and then applies a shared rotation. Expose that
index map through an opt-in return field, leaving old default diagnostics intact,
and permute tree features in both node axes. A rotation changes coordinates but
not these invariant tree features. Failing to transform the tree would make its
labels refer to the wrong atoms. Recheck joint-source invariance and conditional
target marginals; do not borrow the old independent-Gaussian score proxy.

## Bounded test and engineering entry points

- Keep the currently frozen shell source for the first test; do not select a
  different source separately for each seed or tune its parameters.
- Compare no tree context, the actual source tree, and an independently sampled
  sham tree with the same composition-conditioned law. Actual/sham adapters need
  identical parameter counts, initialization and training budgets. This tests
  information about the source component rather than mere extra parameters.
- Implement through a runtime wrapper of `EndpointVectorField.denoise_graph`,
  which receives `g` and embedded `edge_features`. A zero-output-initialized
  residual edge adapter can preserve warm predictions at initialization. The
  shared FlowMol source and bond-supervision weight remain unchanged.
- `prepare_research_backbone` must reconstruct any declared adapter before strict
  checkpoint loading. An enabled adapter must reject missing/mismatched context.
- Extend the FM path with an explicit context-permutation interface inside local
  graph scope. Check transformed features against the actual coupled x0, rotation/
  same-species permutation equivariance, zero-init prediction equality, useful
  adapter gradients and complete checkpoint round-trip behavior.
- Freeze the exact protocol before new generation, use the existing two training
  permutations for a matched initial test, and retain every outcome. New-component
  context must be supplied during generation using the same sampling law.

The main scientific test is a repeatable actual-tree versus sham/no-context
increment. A smaller FM training loss alone is not enough. Source fitting and
transport conditioning should not be changed simultaneously in this first test.
If the increment appears, use a separately frozen composition-generalization panel
and learned-generator comparison. No prior evaluation geometries may enter fit.

The conditional flow's joint density over (T,X) and its marginal density over X
are distinct. A normalized source and a conditional field do not automatically
make the marginal tree-conditioned final density tractable; the current noisy
midpoint sampler stays unqualified for absolute likelihood, ESS and Boltzmann
sampling. Keep thermal claims separate from the generation utility test.
