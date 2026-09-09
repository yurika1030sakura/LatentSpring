# Tempered correction development protocol, September 9

The user explicitly authorizes reconstruction from theory through framework,
with ICLR as the target. This protocol replaces an unsupported density shortcut
with standard, independently checked sampling components; it is not a claim of
new SMC, MALA, Gaussian mixtures, or ICLR readiness.

## Evidence motivating the change

The first eight-atom work pilot (processed validation row 5846) used 32 paths,
32 Euler-Gaussian steps, noise scale .1 and a simple auxiliary backward drift
-v. All energies were finite but ESS was 1.655/32. Across those endpoints, the
proposal log factor has standard deviation 91.90 nats, compared with 12.87 for
the restrained target log value. These sample diagnostics suggest testing the
backward proposal and integration protocol, rather than attributing the entire
failure to the molecular energy landscape. They do not identify a unique cause.

The temperature-corrected alternative uses an independent frozen-FM pilot to
fix M centers in intrinsic COM-free coordinates and defines

    q_S(z) = (1/M) sum_j Normal(z; center_j, sigma^2 I).

This is a normalized, strictly positive finite mixture. Its log density and
score are evaluated by log-sum-exp and mixture responsibilities. q_S is the
new proposal, not the inaccessible FM likelihood. The reference molecular
coordinates do not initialize either the centers or production particles.
The pilot RNG differs from every production RNG. Conditioning on the pilot
allows standard importance/SMC identities; finite-mixture coverage remains an
empirical limitation, especially in high dimension and for rotations.

For fixed composition, charge and declared spin, define
U=E_eSEN+.05 sum_i ||x_i||^2 eV on COM-free H and kT=1 eV.
Finite normalization assumes the neural potential is bounded below; the
harmonic restraint is part of the target, not an unreported sampling trick.
An orthonormal Helmert basis preserves the intrinsic volume element. The
target includes all labeled configurations of that composition, not one
pre-specified bonding topology, and the time parameter is not physical time.

## Fixed-budget comparison

Four arms use the same target and number of potential evaluations:

1. Standard Gaussian prior + random-walk Metropolis.
2. Standard Gaussian prior + Metropolis-adjusted Langevin (MALA).
3. Independent FM-mixture proposal + random-walk Metropolis.
4. The same FM-mixture + MALA.

Initial pilot: 32 particles, 32 fixed stages beta_k=(k/32)^2, one MH move per
stage, proposal standard deviation .08 Angstrom in the orthonormal coordinates,
mixture sigma=.3, and population-score norm cap 100. Every Gaussian reverse
proposal term is included in the MH ratio, including for the clipped score.
All arms query the actual eSEN oracle, through a persistent subprocess in the
separate omol25 environment. Both energy and force costs are included in this
implementation; RWM does not use the returned forces. The cache avoids querying
accepted states twice. Model-center generation cost is reported separately and
is not hidden inside an equal-compute claim.

Work is updated before mutation. If weight ESS falls below N/2, multinomial
resampling occurs and its ancestry is propagated. The temperature schedule
and proposal scale are fixed before the run, not selected using the same
population. Standard SMC then gives an unbiased normalizer estimator under
the usual support/integrability assumptions. Its logarithm and normalized
expectations have finite-particle bias; particles after resampling are not iid.

Report endpoint ESS alongside the number and effective mass of initial
ancestors, acceptance per stage, oracle calls, runtime and geometric spread.
Resetting weights can make ESS equal N even after all particles descend from
one ancestor; such a result does not establish recovered diversity. All oracle
failures are retained; no estimator is constructed by discarding failed calls.

A parallel work-path grid retains the original target, parent and prior seed:
(steps,noise)=(32,.1),(128,.1),(512,.1),(128,.03),(128,.3),(128,1).
This compares weight efficiency of different stochastic proposals. Brownian
paths are not coupled across resolutions, so it is not a per-trajectory
discretization convergence test. Single-seed grids select development settings;
they do not establish a repeated molecular advantage.

## Verification and prior work

140 tests pass against the staged implementation and existing suite. New
checks cover mixture density/score against an independent distribution API
and autodiff, pointwise detailed balance for an asymmetric nonconservative
Gaussian proposal, Gaussian target moments and normalizers with/without
resampling and with RWM/MALA, large energy-offset invariance and force guards.
A real two-environment H2 query succeeds. A four-particle, two-stage molecular
preflight completes 12 oracle calls; endpoint ESS=4 but only two ancestors
survive, illustrating why ancestry reporting is necessary. It is an interface
smoke, not a sampling-quality result.

Required prior art / comparators:

- Neal, AIS: https://arxiv.org/abs/physics/9803008 .
- SNF: https://arxiv.org/abs/2002.06707 .
- FEAT: https://arxiv.org/abs/2504.11516 .
- Flow Perturbation: https://doi.org/10.1038/s41467-025-62039-8 .
- EWFM: https://arxiv.org/abs/2509.03726 .
- FALCON, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/72bcad2c3fe7625fc233fabe84a6d7ef-Abstract-Conference.html .
  It learns few-step flow maps with an invertibility-promoting objective; it
  directly addresses the likelihood cost problem and must not be omitted.
- Prose, NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8780dbea239de8fd1401f9ec6721608d-Abstract-Conference.html .
  Its transferable peptide normalizing flow and importance-based finetuning
  preclude claiming amortization or cross-system transfer alone as new.

Promotion gate: repeated independent molecular benefit with sufficient mixing
and diversity under matched oracle and compute budgets, followed by a clear
contribution beyond these established methods. Raw IDs, electronic-state
metadata and an independent test split remain necessary for broader claims.
