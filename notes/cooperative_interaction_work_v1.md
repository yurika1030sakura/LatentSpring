# Cooperative edits through learned four-state interactions

This is an implemented research prototype, not an established ICLR contribution.
It preserves the FlowMol initializer, OMol25 data conventions, original electronic
conditions and the labelled COM target. Current tests concern terminal exchanges;
they do not generate arbitrary new heavy-atom connectivity or establish an
equilibrium energy distribution.

## The learning problem

For two terminal edits A and B with disjoint moved atoms and passive anchors,
the kinematic maps commute after COM centering. Let their four states be
`x, x_A, x_B, x_AB`. Define the electronic interaction

\[
 I_E=E(x_{AB})-E(x_A)-E(x_B)+E(x).
\]

The target uses the existing inversion-averaged eSEN energy. The known harmonic
term is separate: with COM displacements `d_A=x_A-x`, `d_B=x_B-x`, its interaction
is `I_H=gamma * <d_A,d_B>`. This identity is independently checked. A nonzero
four-state interaction is not a new thermodynamic law, a dissipated-work estimate
or proof of a physical reaction pathway.

All36 original FIT parents were retained. Nine have no eligible independent
pair. On the other27, all eligible pairs were considered and at most32 per parent
selected by a fixed hash before combined-endpoint validity or energy. There are
450 selected pairs,440 valid combined endpoints and10 failures. Source checks
and combined endpoints cost934 new raw oracle calls; the original three corners
reuse hashed, audited labels. Mean absolute electronic interaction, balanced by
parent, is0.16419 eV.24/440 valid pairs reverse the downhill/uphill prediction of
an oracle-additive approximation including exact confinement. Only one has both
single moves uphill and the joint move downhill. This is FIT mechanism evidence,
not a sampler result or an independently selected test set.

## Contrast representation

`InteractionWorkModel` predicts `I_E` using mixed radial contrasts between moved
atoms in the two groups. Gaussian radial functions plus an inverse-distance
feature are evaluated at every corner. Their combination is

\[
 \Phi_{ij}=\phi(r_{ij}^{AB})-\phi(r_{ij}^{A})
             -\phi(r_{ij}^{B})+\phi(r_{ij}^{0}).
\]

Coefficients multiply these contrasts and sum over the four cross-group atom
pairs. The simple control learns an independent radial coefficient for each
element pair. A context-blind neural control shares parameters across continuous
element descriptors. The environment model additionally conditions coefficients
on passive atoms, mean attachment graphs and four-corner midpoint environments.
All coefficient inputs are symmetric under either edit reversal. There is no
claim of a passive graph message-passing encoder or a globally consistent energy
function where neither has been implemented.

Reversing either edit negates the prediction; reversing both preserves it.
Exchanging the two groups, atom relabelling and O(3) transformations preserve it.
The model is zero when either contrast vanishes. Six focused tests and5,280
trained-model reversal checks pass. Both seeds of all three variants have fixed
400-step18/9-parent diagnostic fits and separate27-parent full fits. There is no
checkpoint selection. The earlier18/12 evaluation cohorts and722 reserved
outcomes are absent from fitting. Internal held-parent error is about0.155--0.158
eV for the neural variants,0.167--0.170 for typed radial learning and0.200 for zero
interaction. Environmental conditioning has no established added value over the
context-blind network. Prediction improvement is not a molecular sampling gain.

## Why the interaction is an edge preference

Reversing both edits changes directional total work's sign but leaves `I_E`
unchanged. Thus adding this interaction to an additive directional work predictor
does not produce an antisymmetric total-work model. We instead use it as a
symmetric edge preference. Negative `I_U=I_E+I_H` means the two intermediate
corners have higher average potential than the two diagonal endpoints. It can
identify joint transitions that avoid those intermediate states. It does not
imply that each intermediate individually lies above both endpoints.

The policy multiplies the existing additive-work informed weight by
`exp(tanh(-I_U/0.25 eV))`, and mixes normalized probabilities with10% uniform
mass. Bounds and scale are fixed before molecular outcomes. Controls include
uniform selection, additive work alone, known-restraint interaction alone,
typed radial interaction and context-blind neural interaction. The actual
acceptance test always uses the physical endpoint energy.

## Conditional kernels and random candidate panels

A root block contains four terminal atoms with no element appearing more than
twice. This type-only condition admits two unlike-element pairs and depends on
the terminal set and its elements, which the prescribed exchanges preserve.
Root blocks are drawn uniformly using combinatorial type-count weights. Anchor
or geometric outcomes do not condition this draw. All four corners must pass
the current algorithmic validity and expected-graph checks.

The initial fixed-block experiment averages conditional kernels over those
blocks. Some matchings produce identical unlabelled endpoints: two H/Cl pairs,
for example, may only exchange labels among same-element atoms. That experiment
is preserved, but it cannot alone assess learning which root block to change.

The panel variant draws a set S of root blocks, retains S through the MH test,
and lets the policy choose a complete double edit among all valid matchings in
S. Reverse probabilities are normalized over the SAME panel at the proposed
endpoint. Since terminal membership and elements are unchanged by the move,
the panel draw probabilities at the two endpoints are equal. For an augmented
edit channel `a` with determinant J, the acceptance log ratio is

\[
 -[U(y)-U(x)]/kT+\log|J|
 +\log p(a^{-1}\mid y,S)-\log p(a\mid x,S).
\]

This is ordinary auxiliary-variable MH, with the determinant of the two maps
and actual valid-candidate normalizers. Averaging the conditional kernels over
S preserves the same target. Empty panels give self-loops. The method need not
enumerate all O(N^4) root blocks. The audit verifies inverse maps, opposite
additive works, equal forward/reverse affinities and independent physical
acceptance arithmetic. No source-generator likelihood is inferred.

The panel evaluation reuses the fixed-block experiment's836 raw responses,
matching every query position exactly. It changes the selection kernel, so old
policy outcomes need not match; the physical states and responses must. This
adds zero oracle calls. Both evaluations use the same18 development parents;
three have no eligible double-edit block and remain in the denominator. Results
are one-step evidence inside the cooperative family. A comparison with effective
single-edit chains, full generation and costs remains required for broader claims.

## Related work and remaining claim

[Path Auxiliary Proposal (ICLR2022)](https://iclr.ml.gatech.edu/iclr-2022-papers/)
already composes local moves, including an energy-linearized fast version.
[Informed proposals](https://arxiv.org/abs/1711.07424),
[LSB](https://openreview.net/pdf?id=WEKfdiZYZi-) and
[Delta learning](https://arxiv.org/abs/1503.04987) also precede generic informed
weights, learned adaptation and learned residuals. Composed moves, random-panel
MH, mixed differences and radial functions are not claimed as new general
principles. The candidate contribution is the learned molecular interaction
representation and its useful allocation of corrected cooperative proposals.
That usefulness and the representation's distinction still require evidence;
neither symmetry checks nor this targeted prior-art check establish novelty.

## Completed panel result and next scientific test

Both fixed-block and retained-panel experiments and their full audits are now
complete. Panel mean utility is0.086910 eV for additive linear selection,
0.089929 for typed radial interaction,0.089231 for blind neural interaction,
and0.089108 for environmental interaction. All interaction variants have positive
point differences in both seeds, but the gain is only2--4%, several intervals
cross zero, and environmental conditioning has no added-value result. The prior
strong single-edit linear method gives0.121892 eV on the same18 starting parents.
Thus cooperative interaction is not currently a better energy-descent generator.

A subsequent complete-chain test should predefine exploration/barrier metrics
as well as energy and costs, retain the strong single-edit controls, and use a
small fixed budget. It must not present a change of metric or source state as
confirmation of an earlier unsupported claim. No such chain is running yet.
