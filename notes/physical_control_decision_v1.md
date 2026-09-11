# Physical controls and the next method decision

2026-09-11. The current convex refinement candidate has no established external
advantage. Do not use a cleaner implementation or lower training objective as
evidence that the ICLR method requirement has been met.

Four matched-source MALA/HMC controls complete18048 raw queries each and replay
every proposal and acceptance. MALA uses16 common transitions; HMC uses four
four-step leapfrog trajectories. A fixed random subset of320/512 parents receives
one extra transition, including a single-step HMC tail. Both orientations are
evaluated for every new state. The cache avoids repeated physical evaluations
and preserves rejected-state values. Total production cost is72192 raw queries;
the two176-query engineering checks add352. There are no hidden zero-cost
rejected queries.

MALA projected-energy changes are -4.1013/-4.1189 eV, HMC -4.1998/-4.2050 eV.
Their whole-allocation times are60/64 and65/64 seconds. The same32 prescribed
geometry parents give MALA31/29 and HMC30/29 converged attempts per32, with
success-only strain medians3.0524/2.8456 and2.7689/2.5736 eV. Convex gives
28/29 converged,4.6595/4.6550 eV and139/144 seconds. All four MCMC replicas
are strongly favored over convex by the paired convergence/strain ranking.
This is one development condition, not a general ranking or target-calibration
certificate. Learning amortization on further generated samples remains distinct.

Canonical evidence: `research/evidence/parity_physical_controls_v1.json` and
`research/evidence/parity_all_controls_geometry_v1.json`. The latter retains
all480 attempts across15 arms:423 converged and57 failed, with exact cross-run
source identities and input/energy-log checks. No MCMC endpoint KL, importance
weights, normalizer, or equilibrium ESS is evaluated.

The ideal Metropolized kernels preserve the declared target. For a fixed
target-invariant kernel K and finite relative entropy, the standard Markov
contraction inequality gives KL(qK||pi)<=KL(q||pi). This does not determine the
amount of KL improvement from the present records or prove that a short chain
has reached pi. Outcome-independent compositions/mixtures of such kernels also
preserve pi. Clipped MALA drifts must enter both proposal densities; HMC uses
the actual Hamiltonian in acceptance. Runtime oracle precision is qualified
numerically, not asserted to be exact real arithmetic.

## Prior-art boundary for a possible further reconstruction

[Timewarp (NeurIPS2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/a598c367280f9054434fdcc227ce4d38-Paper-Conference.pdf)
already uses learned conditional flow proposals in Metropolis sampling of
molecular systems. It combines trajectory likelihood training with acceptance
and entropy objectives. Conditional neural proposals plus MH correction or
acceptance training alone are therefore not new ideas.

[Markovian Flow Matching](https://arxiv.org/abs/2405.14392) already combines
flow-matching training with local/nonlocal MCMC and adaptive tempering. Merely
adding FM to a Markov chain is not a defensible new-method claim either.

Before another neural branch, identify a measured distributional bottleneck
and a precise intervention that existing controls do not solve efficiently.
Use bounded independent-reference diagnostics, then test the intervention with
matched physical cost and invariant mode-sensitive observables. Do not use
global translation/rotation or atom relabelling as evidence of physical mixing.

Eight raw development reference geometries and electronic states already exist
in `runs/development_fm_baseline_v1/references.json`, linked to the frozen
development manifest. They may seed independent reference assessments, not the
generator. A raw QC geometry is not an equilibrium ensemble. References must
retain different initialization families and demonstrate adequate mixing and
uncertainty before any claim about Boltzmann populations.

The main48-arm campaign still needs completion; reserved722 outcomes stay
untouched. No large capacity grid or further claim polishing is justified by
the current learned-refiner results.
