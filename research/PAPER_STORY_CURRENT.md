# Current paper direction — September10,2026

Working title: **Learning Molecular Geometry Distributions with Nonequilibrium Work**.
This is a provisional research direction. The existing main text remains an
inherited-method audit, with later development experiments in the appendix.
It is not a qualified ICLR method paper. The latest build has9 main pages at
runs/verification/paper_20260910_species_replication/main.pdf; scientific readiness is false.

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
The subsequent learned-feature correction also improved measured DSM but failed
moment qualification. No score-based actor was released; this recipe is stopped.


## Exact-entropy refinement direction

A separate invertible linear adapter now demonstrates a small relative-KL
improvement on a fresh1024-sample panel, without estimating the base score:
typed -.10193+/-.02719 nat, scalar -.06356+/-.01809. This is one trained seed and
one condition, with poor remaining path ESS. The work/volume accounting is exact
for the stated finite map; the objective is established normalizing-flow theory.

The complete nonlinear species-coupling adapter is now implemented, including
invariant neural contexts, internal-element-group maps and element-centroid
couplings. Each layer has exact constrained volume and a checked inverse.
It learns geometry-dependent nonlinear transformations from physical forces
and exact entropy change, while the FlowMol base stays frozen.

On one eight-atom condition, two1000-step training pairs show small advantages
over matched typed-linear controls: -.04892+/-.01732 and-.03882+/-.01642 nat
on the shared2048-row fresh evaluation panel. This is training replication
with common parent pools and evaluation noise. All arms remain path-weight
degenerate, ESS about1/2048. The new code is an implemented method candidate;
these data do not qualify a general Boltzmann sampler or establish AI novelty.

The candidate contribution to investigate is symmetry-preserving molecular
refinement with inexpensive exact COM-volume accounting and geometry-dependent
neural expressivity. EACF, equivariant finite flows, convex potential flows and
residual flows are direct prior art. The KL and determinant identities are not
new. Strong coupling-flow controls, more molecular conditions and full cost
comparisons are required before presenting a distinct ICLR contribution.

The development appendix now includes the complete architecture, assumptions,
replicated outcomes and retained ESS failure. The main text remains an audit
and must be rewritten around a qualified contribution when the evidence warrants
it. See notes/species_coupling_execution_v1.md and
research/evidence/species_entropy_replication_v1.json.
