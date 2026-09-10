# Longer-chain controls and fixed-observation path scoring

Frozen before the first long-chain results, September 10 UTC, 2026.

The AgBr2 300 K target retains the explicit 0.1 eV/A2 COM restraint and original
charge 0 / multiplicity 2. Both MCMC controls choose eight starts uniformly
without replacement from the same 64 FM16 samples (selection seed 9062), without
ranking energy or geometry. Each spends 8,512 potential queries including initial
states and rejected proposals. MALA makes 1,063 moves per chain with standard
deviation 0.1 sqrt(kT). HMC refreshes momenta for trajectories with leapfrog
counts [7] + [32]*33, step 0.05 sqrt(kT). Both cap score norms at 100/kT and use
true Metropolis corrections; HMC uses true target plus momentum energies.

Both collect at force-update counts 615, 679, 743, 807, 871, 935, 999, 1063.
The resulting 64 samples have only eight independent chain clusters. Comparisons
use chain-clustered uncertainty, exclude unknown initialization bias and do not
claim equilibrium, endpoint density, importance ESS or a normalizer. xTB uses
32 predetermined indices (seed 9059), retaining failures and cluster identifiers.
These are prespecified stronger controls, not tuned or converged gold standards.

Fresh frozen-proposal evaluations use 4,096 independent paths, batch 64, seed
9065 for all three annealed-joint seeds and both fixed-joint/energy controls.
The source-stream loader check passed before launching them. All comparisons
use independent reference v2; reference uncertainty remains explicit. Report
additional inference queries separately from the 8,512-query training protocol.

`cfm_mol/path_balance.py` separately supplies normalized path probability factors
for detached observed trajectories. It batches all time slices, requires rowwise
independent drift callbacks, and computes parameter scores at fixed observations,
not sampling-path derivatives or endpoint densities. Its per-condition variance
loss is a prior-art baseline primitive (Richter and Berner, Improved sampling via
learned diffusions, arXiv:2307.01198); no molecular benefit has been tested. Values
and finite-difference gradients match the finite-path sampler on analytic fields.
Real FlowMol time-batch/gradient agreement is required before molecular use.

Validation: 216 tests pass, including HMC target-moment preservation under clipped
kicks, exact query counts, cluster uncertainty and fixed-path score gradients.

## Real-interface gate and molecular log-variance pilot

The real electronically conditioned CTMCVectorField test now passes: six distinct
times including both endpoints yield matching independent versus batched outputs
and parameter gradients; full observed-path factors also match. Self-conditioning,
model modes and graph features are restored. Complete suite: 217 tests.

The optional `--objective log_variance` branch draws fresh current-forward paths
without gradients, evaluates their terminal energies once, then differentiates
variance of U/kT + log q0 + log K - log L with states and energies fixed. No replay,
resampling, reference positions, endpoint force gradients or normalizer fitting is
used. Each step checks observed-path versus sampler work within 0.05 nat (mixed
float32 backbone / float64 factors); failures abort. Record all errors. This is
a discrete normalized-kernel baseline motivated by Richter and Berner (ICLR 2024),
not their exact continuous-time implementation and not a new loss. See
https://arxiv.org/abs/2307.01198, Section 2.3, for the fixed-reference measure.

First production gate: eight-atom row 5846, 300 K, native mean, annealing exponent
0.5, two updates, batch two, 16 transitions, 64 evaluation particles before and
after, seed 9066. The 132-query run verifies execution only. A matched 500-update
pilot is conditional on finite gradients, factor agreement and feasible memory.

## Prespecified second evaluation stream

The first 4096-path stream yielded low normalizer estimates for all three trained
seeds, with paired weight influence correlations 0.40--0.67 and 17--24% of mass
in the largest 1% of weights. These correlated estimates do not independently
confirm a common bias. Before new draws, freeze seed 9067 as a second 4096-path
stream, batch 64, for the same three proposals and both training controls. Keep
all first-stream results. Reference v2 stays fixed. This adds 20,480 evaluation
queries and no training; assess between-stream variability, without selecting
the stream nearest the reference or interpreting nested prefixes as replicates.

## Eight-atom matched HMC control

The batch-16 log-variance execution gate passed with peak GPU allocation 4.44 GiB
and maximum path-work discrepancy 2.24e-4 nat. The 500-update, batch-16 run
45760898 starts from the same FM checkpoint and uses seed 9051, 256 before/after
paths and 8,512 total oracle queries, matching the earlier joint/energy runs.
Its two-update predecessors are execution gates and provide no efficiency claim.

Extend the prespecified AgBr2 HMC recipe unchanged to the original eight-atom
case, using work_fm_control_5846_v2 FM16 starts. Keep the same random selection,
noise seed, 8,512-query budget, collection times and xTB indices. No independent
eight-atom normalizer reference exists; omit that comparison explicitly. This
control evaluates geometry and finite-chain behavior, not equilibrium truth.
