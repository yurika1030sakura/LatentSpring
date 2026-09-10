# Current paper direction — September10,2026

Working title: **Learning Molecular Geometry Distributions with Nonequilibrium Work**.
This is a provisional research direction. The existing main text remains an
inherited-method audit, with later development experiments in the appendix.
It is not a qualified ICLR method paper. The latest build has9 main pages at
runs/verification/paper_20260910_calibration/main.pdf; scientific readiness is false.

## Scientific question and actual scope

Can a pretrained molecular flow generate useful conditional3D configurations
with quantitatively reliable probabilities under a specified energy target,
at a useful total cost relative to strong sampling methods? Geometry, energy,
and probability mass are separate requirements.

Condition on atomic identities/composition, charge and spin. The target on the
labelled COM-free coordinate space is proportional to exp[-U/(kT)], with
U=E_eSEN+.05 sum_i||x_i||^2, at the stated temperature. The restraint and ML
potential are part of the target; this is not an unconfined experimental or
DFT ensemble. Artificial flow time is not molecular-dynamics time. An energy
histogram includes density-of-states effects, not just exp(-E/kT).

The implemented Gaussian-path branch has explicit forward/backward factors.
Known nonequilibrium importance identities do not establish finite-sample
coverage. Corrected CNF density remains numerically unresolved. The later
endpoint-entropy gradient is implemented and verified on analytic examples,
but its molecular proposal-score estimators have not qualified; no molecular
actor update has occurred under this prototype.

## What the evidence supports

| Question | Current finding |
|---|---|
| Conditional3D generation | Implemented with audited electronic metadata and exercised on a frozen eight-condition development panel. |
| Geometric utility | Some panels have high xTB convergence and lower strain. Relaxation success is not complete chemical validity, correct bonding, connectedness or synthesizability. |
| Distribution checks | Repeated weighted AgBr2 statistics have an independent numerical reference and same-box cubature check; this is only three atoms, with finite uncertainty and no global coverage certificate. |
| General Boltzmann sampling | Not established. Eight-atom work, covariance and empirical-CFM variants remain weight-degenerate. |
| Global auxiliary repair | A normalized innovation-coordinate Gaussian diagnostic preserves endpoints but fails its32-path screen. |
| Endpoint-entropy learning | Scalar mechanism works; the molecular critic's initial small-panel pass fails larger independent confirmation, a second seed, and longer matched training. |
| Unweighted target samples | Not qualified. Reweighting/resampling does not create independent information or certify raw outputs. |
| HMC superiority | Not established. Budget8512-query HMC and25024-query HMC must not be conflated; retained MCMC samples are correlated. |
| Fully unconditional new composition/charge/spin generation | Not qualified by these conditional experiments. |

## Contribution boundary and next research decision

Reusing FlowMol, invoking AFM/Jarzynski, adding energy losses, Gaussian path
weights, annealing, importance-weighted CFM, global Gaussian conditioning or
score-difference updates does not by itself establish novelty. Relevant direct
prior work includes SNF, FEAT, EWFM, MFM, flow perturbation, VSD/DMD and NDSM
control variates. See the primary-source audit in
notes/forward_mass_update_candidate.md and notes/endpoint_entropy_candidate.md.

The preferred contribution remains a learning intervention that resolves a
measured probability-mass or inference-error problem, with a clear mechanism
and replicated benefits. The latest score failures require separating finite
parent-pool effects, optimization noise and score-representation error before
another recipe is promoted. The closed recipes are stopped; no unreliable
critic should update the molecular generator.

An ICLR claim still requires a defensible contribution, numerical consistency,
multiple independently selected molecular conditions and training seeds, credible
distribution checks, and total-compute comparisons. The reserved722 conditions
remain untouched. Use STATUS.md, NEXT.md and the complete evidence artifacts;
never replace failed confirmations with the earlier selected passing screen.


## Current concrete method candidate

Residual-calibrated endpoint-entropy learning is now specified, and its frozen
calibration component is implemented. A constrained Stein projection preserves
symmetry and normalizability and has a standard population risk-reduction
property. In16384 independent samples, it improves a measurable component of
score error for both neural seeds and beats a calibrated Gaussian baseline.
However, strong unfitted radial/angular violations remain; no generator update
is qualified. This is a partial component result, not a full method contribution.
See notes/stein_calibrated_entropy_candidate.md and score_calibration_audit_v1.json.
The next unimplemented candidate is correction in learned invariant feature
directions, with independent validation and direct Stein-estimation baselines.


## Exact-entropy refinement direction

A separate invertible linear adapter now demonstrates a small relative-KL
improvement on a fresh1024-sample panel, without estimating the base score:
typed -.10193+/-.02719 nat, scalar -.06356+/-.01809. This is one trained seed and
one condition, with poor remaining path ESS. The work/volume accounting is exact
for the stated finite map; the objective is established normalizing-flow theory.

The next architectural candidate is nonlinear species coupling with exact
COM-constrained volume. Its centered convex point-map primitive is implemented
and unit-tested; neural contexts and complete molecular couplings remain to be
implemented. This design seeks more flexible refinement while avoiding the
failed score-estimation dependency. EACF, equivariant finite flows, convex
potential flows and residual flows are direct prior art. No novelty or nonlinear
sampling result is claimed. See notes/species_coupling_adapter_candidate.md.
