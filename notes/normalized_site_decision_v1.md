# Normalized site-guide decision — September12, 2026

The ICLR goal is ACTIVE and unachieved. New model code, four training arms,
eight sampling arms, the independent probability audit and the17-atom support
screen are complete. The new work is an AI method candidate, not an established
ICLR novelty or sampling-superiority claim.

## Implemented method

`cfm_mol/normalized_site_guide.py` uses masked geometry, graph/electronic context
and continuous atomic-number/radius descriptors to predict a normalized vMF
mixture. A learned equivariant center corrects the physical site prior; paired
offsets and weights are indexed by passive atoms. The actual normalized
directional surface score is trained. A vector ablation keeps the corrected
center. Total parameters4531; vector active parameters4497.

`defensive_joint_proposal` in `cfm_mol/joint_chemical_geometry.py` mixes whole
physical and learned joint proposals with equal weights. Both complete densities
are evaluated in BOTH directions. Root radii use the Cartesian r^-3 factor;
the augmented root order and reverse templates are retained. The accepted
off-diagonal flow is at least half the physical component's flow for the same
action/order. This standard mixture bound says nothing about training cost,
wall-clock advantage or finite-time equilibrium accuracy.

Fourteen targeted tests passed: mixture normalization, analytic surface score,
zero-concentration gradients, sampling moments, O(3)/permutation covariance,
masked-context and batch invariance, joint inverse/volume and the physical-flow
bound. These are implementation checks, not an all-repository or scientific PASS.

## Completed evidence

- Training array46114318: all four tasks completed. Only the original generated
  training table was used. No new physical training calls. Initial and final
  force errors were independently recomputed; the optimizer trajectory was not
  independently replayed.
- Training MSE: physical prior708.095; vector558.790/545.641;
  mixture571.552/549.580. More mixture components do not improve this fit.
- Sampling46114697: all eight arms complete,14820 new raw physical queries.
  CPU audit46114876 replays every state, raw inversion pair, candidate, RNG and
  acceptance. Independent math checks the normalized components, sequential
  contexts and complete marginal mixture. Maximum coordinate replay error0.

| Method | Graph hits among4 starts, replicas0/1 | Joint accepts of256 | Total raw queries |
|---|---|---|---|
| Site |3 /2 |4 /3 |2734 /2756 |
| Untrained mixture |2 /3 |3 /4 |2734 /2750 |
| Learned vector |4 /4 |5 /5 |12446 /12456 |
| Learned mixture |4 /4 |5 /5 |12460 /12452 |

All-four first passages: vector steps127/55 at total costs11496/10982;
mixture215/55 at12158/10982. These remain expensive four-parent results. The
earlier full-cost site controls reached all four for6380/2994. They use other
transition seeds, so do not assert a paired statistical comparison; they still
preclude treating a short-pilot win as demonstrated full-cost superiority.

Final-half potential Rhat: vector1.950/1.658, mixture1.677/1.808. Equilibrium and
energy-distribution qualification remain absent. Every study still uses the same
four repeatedly inspected development starts; there is no independent molecular
generalization claim.

The frozen17-atom screen reuses only cached TRAINING coordinates from32 parents,
with four proposals per parent. No physical queries or new fitting occur.
Site validity101/128 and96/128 is reproduced exactly. Initialization93/94,
learned vector92/85 and learned mixture99/86 show that the old zero-validity
failure is absent, but the support improvement already appears without learning.
Independent forward/reverse density checks pass on supported endpoints. This is
not an energy or molecular sampling transfer experiment.

## Novelty and next work

Read `notes/normalized_site_novelty_review.md`. VonMisesNet already uses learned
directional mixtures for molecular conformations; MARS covers learned graph-edit
proposals, and equivariant VFM covers symmetry-aware post-hoc control. Our
candidate combination must earn its contribution through useful joint
connectivity/geometry moves under a specified physical target and strong evidence.

Stop enlarging the current singleton mixture: it has not beaten the vector
ablation or the strong physical prior after preparation. Next implement the
involutive pendant-fragment lift and its full augmented Jacobian tests in
`notes/fragment_exchange_design.md`, then assess coverage on independent molecular
conditions before fragment-specific learning. The same note specifies a separate
geometry-only source/preparation repair with fresh seeds and explicit new costs.
Do not erase previously paid source-energy evaluations from historical ledgers.

Current manuscript: `paper/angular_working.tex`, seven main-text pages with the
new method/proofs, completed positive/negative evidence and close prior art.
The original manuscript remains unchanged at `paper/main.tex`. Scientific
submission readiness is false, and reserved evaluation outcomes remain untouched.
