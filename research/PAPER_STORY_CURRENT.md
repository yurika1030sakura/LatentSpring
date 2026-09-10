# Current paper direction — September 10, 2026 UTC

Working title: **Learning Molecular Geometry Distributions with Nonequilibrium Work**.
This is a research direction, not a validated new-method claim. The existing PDF
main text remains an audit/development manuscript about the inherited local
readout. New sampling results are in its development appendix. A final ICLR
method manuscript still needs a defensible contribution and stronger evidence.

## Main question

Can a pretrained molecular flow be adapted to generate geometrically useful
configurations with quantitatively checked probabilities under a specified
energy-based target, at a useful cost relative to strong sampling baselines?
Lower energy, successful geometry relaxation, and correct probability mass are
separate outcomes. An energy minimizer can produce a narrow, low-energy cloud
while representing the target distribution poorly.

## Actual method and scope

Condition on atom identities/composition, total charge and spin multiplicity.
The repaired position-only FlowMol backbone supplies the starting geometry
model. The active physics branch trains forward and auxiliary backward Gaussian
path kernels using mean generalized work and exact discrete path factors.
External eSEN energies and forces provide the physical signal; COM-free
orthonormal coordinates and the actual sampling kernel define the measure.
The eventual distilled flow-matching student has not been qualified.

The target is p(x|c) proportional to exp[-U(x,c)/(kT)] on the zero-centroid
coordinate space, where U=E_eSEN + .05 sum_i ||x_i||^2 in eV. The added restraint
is part of the target. This is not an unconfined gas-phase or experimentally
validated DFT ensemble. The artificial path time is not molecular-dynamics time.

Current positive distribution checks concern **importance-weighted outputs**.
They do not prove that raw, equally weighted neural outputs already follow the
target. Reweighting needs sufficient coverage and finite-sample efficiency;
resampling alone does not create independent information. Also, a histogram
of total energy is proportional to its density of states times exp(-U/kT), not
to exp(-U/kT) alone. The bare eSEN-energy histogram additionally reflects the
restraint and the available configurations at that energy.

## What is and is not supported

| Claim | Current evidence |
|---|---|
| Conditional 3D coordinate generation | Implemented and exercised with original charge/spin, including eight new development conditions. |
| Geometrically useful output | Some generated panels have high xTB convergence and improved strain; convergence is not full chemical validity, connectedness, bonding correctness or synthesizability. |
| Calibrated small-system statistics | AgBr2 has three trained seeds, two fresh streams each, and independent quadrature plus a direct same-box cross-check; weighted statistics have support within stated uncertainty, without a global convergence certificate. |
| Broad Boltzmann sampling | Not established. The completed eight-atom 500-step arms have ESS near one; 1500-step continuations remain live. |
| Noise annealing advantage | Not established: fixed-noise AgBr2 joint control is competitive. |
| Superiority over HMC | Not established. Matched-query HMC is a substantive control and uses much less allocation time in the completed small runs. |
| Fully unconditional generation of new composition/charge/spin and geometry | Not qualified by the current conditional experiments. |
| Fast unweighted distilled FM generator with target statistics | Not established. |

## AI novelty position

Reusing FlowMol, adding energy supervision, invoking Jarzynski/AFM ideas,
training forward/backward stochastic kernels, or correcting path weights are
not sufficient novelty. SNF already combines trainable transport and stochastic
blocks with importance weights: https://arxiv.org/abs/2002.06707.
FEAT develops neural nonequilibrium free-energy estimators with learned
transport: https://arxiv.org/abs/2504.11516. Energy-weighted flow matching is
also established: https://arxiv.org/abs/2509.03726.

The possible contribution lies in a learning/sampling algorithm that resolves
the measured stiffness, optimization and weight-degeneracy problems for
conditional molecular geometry, with predictable and replicated gains in
calibration and compute. The Gaussian noise/ESS limit is a quantitative
analysis tool, not yet a sufficient standalone contribution. A new algorithm
and causal ablations supporting that contribution have not been established.

If the eventual evidence supports a method paper, its narrative should link:
(1) a precisely measured failure of existing approaches; (2) a distinct,
principled learning intervention; (3) calibrated distributional and geometric
benefits across independently selected molecular conditions; and (4) a fair
compute comparison. Current repairs and small-system results supply parts of
this chain. They do not justify writing the remaining parts as completed work.

See STATUS.md, NEXT.md and the immutable evidence files for quantitative results.
