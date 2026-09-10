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
