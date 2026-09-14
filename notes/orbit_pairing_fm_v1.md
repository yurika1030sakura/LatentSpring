# Symmetry and collision-aware pairing in the main FM generator

This is a prospective, bounded generator experiment. The previous latent-mass
module only changed calibration covariance and could not repair marginal weight
collapse or conditional shapes. This experiment changes the main FM training
coupling and evaluates fresh molecular outputs. It is not established ICLR
novelty, not a new physical law, and not an equilibrium-generation result.

## Pairing and its marginal laws

Draw a centered isotropic Gaussian X0 and an OMol25 training geometry X1, with
the original atomic labels, global charge and spin. Choose a proper rotation
R(X0,X1,A) of X0, where A is independent candidate-search randomness. Then draw
an independent Haar rotation H on SO(3) and set

    X0' = X0 R H,      X1' = X1 H.

The norm and SO(3) orbit of X0 are unchanged by R. Conditional on X0,X1,A,
R H is Haar, regardless of how R was selected. Hence X0' has the same centered
Gaussian law as X0. Conditional on X1, X1 H is uniformly rotated data. Molecular
shape and chirality are unchanged. The endpoints are generally correlated.

The FM interpolation remains Xt=(1-alpha(t)/alpha(T))X0' +
alpha(t)/alpha(T)X1', with its actual displacement derivative as the regression
target. Therefore the population conditional-regression argument applies to the
Gaussian source and rotation-symmetrized empirical data endpoint. This does not
require assigning a density or Jacobian to the noninvertible pairing procedure.
Inference is still the existing memoryless composition-clamped ODE.

This construction licenses rotation alignment for the declared ambient Gaussian
prior. It does not prove that the earlier decision to use independent coupling
was mathematically wrong: independent coupling is valid, but is not the only
valid option. Restoring alignment may or may not improve finite-model training.
We do not modify existing inference/density functions or relabel archived samples.

## Three matched training arms

All arms share an independent Gaussian/time stream and a separate, shared stream
for candidate rotations and Haar augmentation. The methods are:

1. Independent pairing, with the same shared Haar augmentation as the others.
2. Proper Kabsch rotation minimizing squared positional displacement. There is
   no atom permutation search in this baseline; do not call it full equivariant OT.
3. Collision-aware rotation search. Include the Kabsch result and twelve proper
   perturbations about six random axes, at plus/minus0.4 or0.8 radians. Select the
   minimum of mean squared endpoint displacement plus4 Angstrom^2 times a
   dimensionless overlap cost at interpolation times0.25,0.5,0.75.

For pair ij the overlap contribution is
max(0,1-d_ij^2/[0.6(r_i+r_j)]^2)^2. Sum over unordered pairs, average over the
three slices and divide by atom count. Radii are ordinary covalent radii. This is
a geometric training cost, not an electronic energy, chemical graph, bond label,
Boltzmann weight or guarantee of a collision-free path. The finite candidate
search is not claimed to find an optimal group coupling. A reduced training-path
overlap cost is not itself a molecular-generation improvement.

No scale change is allowed in the pairing, since arbitrary data-dependent scaling
would alter the Gaussian radial distribution and is not repaired by Haar rotation.
No reflections are introduced. Atoms and their metadata retain their ordering.
The final shared Haar rotation also avoids claiming that a deterministic arbitrary
gauge choice has the original Gaussian distribution before symmetrization.

## Prior art and contribution boundary

[Equivariant flow matching, NeurIPS2023](https://arxiv.org/abs/2306.15030)
already uses symmetry-aware transport costs and rotation/permutation alignment.
[ET-Flow](https://arxiv.org/abs/2410.22388) uses rotational alignment for molecular
conformer generation. [SemlaFlow](https://arxiv.org/abs/2406.07266) extends molecular
equivariant transport with scale optimization. These are direct baselines for
claims about faster/easier molecular FM paths; alignment itself is not new.

Physics-aware probability paths are also not an unexplored general idea:
[PFM submission](https://openreview.net/pdf?id=CEuzrRs613) incorporates type/position
and pocket information and geometric/steric objectives, while
[GO-Flow](https://arxiv.org/abs/2605.25577) proposes a molecular manifold decomposition.
This narrow search is not a novelty certificate. Our current test is whether a
simple collision-aware symmetry pairing adds measurable value over standard
rotation in this bond-free OMol25 generator. Even a positive result would need
replication, a sharper contribution and broader relevant learned baselines.

## Frozen first molecular experiment

`research/evidence/orbit_pairing_protocol_v1.json` records the complete fixed
protocol. The warm model is the verified electronic-conditioned displacement FM,
SHA256 aa25f1c31a7f5d692a8b6a0d1ccb8f564dc82ead95be05c5419f157ba63bbf18.
All three arms take3000 AdamW updates from exactly that model, learning rate2e-5,
batch1, seed29401, with the same real OMol25 training order. No generated evaluation
parent or reserved outcome is a training example. These are FM-only continuation
controls; the original BGFM hook/loss remains intact.

The existing independent-Gaussian interpolation velocity-to-score identity does
not apply to these correlated endpoints. Do not silently enable the legacy
force proxy on this coupling. Any future energy training must use a compatible
score or the actual endpoint-density/energy objective, with its estimator limits.

Evaluate the frozen warm model and all three final models on all eight existing
development compositions,64 fresh Gaussian/noise draws per condition, with common
random numbers. Use the existing64-step midpoint T1 displacement sampler plus
0.025-A COM noise. This finite sampler has no qualified absolute density. Each
model produces512 outputs,2048 total; no reference coordinates initialize them.

Report every attempted output, geometric support, RDKit graph support, validator
errors, distinct constitutional connectivity, size and actual generation/training
time. These fixed-composition generations are not the native joint FlowMol3
sampler. Graph perception remains an algorithmic validity readout, not a quantum
certificate. One training seed is a feasibility test, not a replicated result.

The energy follow-up is conditional and frozen in advance: only if the steric
arm's pooled graph-supported count exceeds BOTH independent and rotation, score
all four methods' same fixed outputs with E_plus and the0.1-eV/A^2 COM restraint,
retaining all conditions and failures. Otherwise stop this bounded pilot without
new molecular oracle calls. This operational screen does not establish statistical
superiority and cannot replace a later independent confirmation if positive.

## Checks and current execution

The new primitive, optional FM path and existing density/condition tests pass:
44 tests initially, followed by5 focused tests after adding an explicit batched
path check. They cover proper rotations, shape/COM preservation, the selection
cost, Gaussian marginal moments after severe alignment, endpoint derivative
semantics and protection of the input graph. The marginal-law proof above is
needed in addition to finite-sample moment checks.

GPU scheduling required preserving two cancelled pending jobs and one rejected
submission. The test partition permits only two submitted jobs per user; a
three-element array exceeds that limit even with concurrency capped at two.
The replacement runs the three arms sequentially in one bounded GPU allocation.
See `research/ORBIT_PAIRING_STATE_20260913.json` for current handles and exact
next commands. Do not modify the frozen scientific protocol to fix scheduling.
