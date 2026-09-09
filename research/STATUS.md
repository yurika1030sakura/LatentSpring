# Active research status — September 9, 2026

**Not yet submission ready.** The project now has a specified conditional
sampler, full density gradients and independent generation checks. No molecular
benefit of the corrected energy objective has been established. Numerical
convergence remains a blocking scientific gate, not merely a runtime issue.
Authors and submission accounts are outside the user's requested execution.

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-research`. Frozen original audit: `../audits/bgfm_20260908`.
Every submitted job uses a committed source snapshot; `jobs.jsonl` also records
rejected submissions. A scheduler state of COMPLETED alone does not certify
that all scientific checks passed.

## What is established

- 111 tests pass, including real FlowMol parameter gradients, full-state and
  prior differentiation, exact discrete-adjoint comparisons, COM density,
  conditional FM targets, stochastic-replica identities and smooth geometry.
- Legacy scalar ordering cannot be promoted to likelihood or Boltzmann sampling.
  The old routine integrates an endpoint prediction as velocity and freezes
  trajectory gradients. It is retained only for historical reproduction.
- The conditional factorization is `pi_frozen(m) q_theta,T(x|m)`. The positional
  checkpoint has no history, prior alignment or sampling retractions. Its
  changed shared backbone must not supply the frozen composition factor.
- Off-policy log-dispersion and independent-product identities are prior work
  or standard mathematics. They are not sufficient novelty. See
  `notes/literature_positioning.md` and `notes/jacobian_noise_bias.md`.

## Completed experiments

| Experiment | Result | Run / job |
|---|---|---|
| Original corrected-energy smoke | 20 updates, 20 finite energy contributions, 5.38 GiB peak | `runtime_smoke_v3`, 45570868 |
| Equal two-trace-budget controls | Squared/product × independent/common, 20 updates each, finite; independent arms skip one outlier each | `estimator_smokes_v1`, 45572136 |
| Affine Gaussian mechanism, five seeds | Mean exact KL: squared independent 0.74756; product 0.00138; common, exact and rotated controls reach zero | `toy_trace_v1`, 45572134 |
| Nonlinear shear mechanism, five seeds | Mean exact KL: squared Rad. 0.08547; product Rad. 0.00259; common squared 0.08141; common product 0.03745 with substantial variance | `toy_nonlinear_v1`, 45578222 |
| Original density resolution panel | Two of eight parents drift by 90.36 and 66.18 nats at 64→128 steps | `numerical_panel_v1`, 45569978 |
| Independent adaptive reference | Easy case succeeds at two tolerances; midpoint-128 differs by 0.2792 nats; both hard cases exceed 6,000 evaluations at both tolerances | `adaptive_panel_v3`, 45577793 |
| Conditional FM, endpoint head, T=0.8 | 1,000 updates, finite, 104.2 s, 5.06 GiB | `position_cfm_v1`, 45575822 |
| Conditional FM, endpoint head, T=0.8 | 10,000 updates, finite, 1,036.1 s, 5.11 GiB | `position_cfm_10000_s9004_v2`, 45580289 |
| Independent xTB, warm / 1,000 FM updates | 29/32 / 30/32 relaxations converge; successful-only median strain 13.09 / 5.01 eV | `position_xtb_v1`, 45577833 |
| Independent xTB, 10,000 FM updates | 31/32 converge; successful-only median strain 4.49 eV | `position_assessment_10000_v1`, 45582154 |
| 1,000-update density, full five sigmas | Two of eight parents still drift 82.74 and 112.61 nats at 64→128 | `position_density_v1`, 45577827 |
| 10,000-update density, sigmas 0.03/0.06 | Two of eight parents fail 0.1-nat gate at 128→256: 0.2561 and 0.4563 nats | `local_density_mature_v2`, 45582155 |
| Smoothed geometry counterfactual, rho=0.1 | Full-panel worst 64→128 drift falls to 2.5136 nats, still not converged; this changes the field without retraining | `smooth_density_v3`, 45584305 |
| Checkpointed high-resolution energy | Four updates, 128 steps, two local siblings, 609 s, 19.03 GiB | `highres_energy_smoke_v1`, 45581074 |
| Exact discrete-adjoint high-resolution energy | Same four updates, 275 s, 1.69 GiB; relative L2 final-weight difference 3.18e-8 from checkpointed run | `adjoint_energy_smoke_v1`, 45583400 |

All molecular training above uses a single seed; warm versus 1,000 versus
10,000 updates is not seed replication. Warm q_0.8 was not trained for the new
conditional path, so its sampling comparison is an initialization diagnostic,
not a fair comparison with the original joint generator. The same eight
conditions and four priors per condition are reused. Reference structures
converge 8/8 with median strain 1.07 eV. xTB uses the recorded interior charge
and declared minimum electron-parity spin; the original DFT spin is unavailable.
Failure-conditioned medians alone do not establish quality, connectivity,
diversity or Boltzmann populations. The 10,000-step sampler's maximum
64→128 coordinate RMS drift is 0.0930 Angstrom.

The adjoint computes the exact first parameter gradient of the discretized
objective by saving states and recomputing each step, rather than inverting the
trajectory. It is an engineering improvement, not a new adjoint theorem. It
does not support arbitrary higher parameter derivatives. GPU agreement is
within floating-point error, not bitwise equality. See
`evidence/adjoint_comparison.json` and `notes/discrete_adjoint.md`.

The numerical panel uses probes fixed across time and solver resolutions;
training refreshes probes across steps. Noise estimates cannot be transferred
between these policies. Replica products correct stochastic squared-loss bias
on a fixed trajectory, not quadrature bias. Common probes are a required
control, not an omitted competitor. All Gaussian and unstable product toy
arms remain in `evidence/evidence.json` and the development appendix.

## New provenance and theory checks

Deterministic replay exactly reproduces all 150,000 training and 50,000 validation
perturbed geometries and atom/charge labels. In these two shards, perturbation
parent index equals the processed source row index. This recovers processed-tensor
provenance and the Gaussian perturbation mechanism, not raw OMol identifiers,
spin, unclipped charge or energy precision. See
`evidence/train_perturbation_provenance.json` and
`evidence/val_perturbation_provenance.json`.

The follow-up theory audit removes unsupported gradient-noise lower bounds,
corrects the endpoint-force conditional expectation, distinguishes discrete
quadrature from an exactly normalized discrete sampler, and replaces the old
capability matrix with the actual sampler/density scope. See
`audit/20260909/THEORY_FOLLOWUP.md`.

Classical RK4 is now available for both sampling and density, including full
discrete-adjoint gradients. It passes analytic fourth-order and real-network
gradient checks. Its extra trace evaluations are explicitly counted; it is not
an equal-cost replacement per step. Molecular RK4 experiments have not yet run.
See `notes/rk4_density_protocol.md`.

## Running, with automatic evaluation

Source `b6921f0e27c6f8df6a93b77a953e757ec04527b9`:

- 45586851, `displacement_development_rho0_v1`: direct residual velocity head,
  T=1, original geometric normalization, 10,000 FM-only updates.
- 45586884, `displacement_development_rho01_v2`: identical initialization,
  training order and seed, smooth geometric normalization rho=0.1 Angstrom.

Each job automatically follows training with matched conditional sampling,
independent xTB evaluation and the eight-parent local density panel at
64/128/256 steps. No energy advantage is claimed before those outputs exist.
This head removes the endpoint conversion denominator but does not guarantee
a well-conditioned learned flow. The full target and predeclared comparison
are in `notes/displacement_geometry_flow.md`.

## Data and evidence limits

A full composition census scans 3,902,107 archived training structures. There
are 6,527 validation structures with absent training stoichiometries, including
59 with at most 12 atoms and 304 with at most 24. Perturbation shards contain
14 and 70 matching parents at these limits. This is a conservative composition-
disjoint development pool, not a new blind test set. The old test file exactly
duplicates validation. Energy labels have already lost precision in float32;
charge clipping, missing source IDs and missing spin cannot be repaired from
those tensors. New loader code preserves float64 when supplied.

Raw directories are empty. The official Hugging Face **model repository**
`facebook/OMol25` reports manual gating. A request for authentication on the
machine or a raw-data backup is pending; credentials should not be sent in
chat. The two environments remain separate. No home-directory writes or
shared FlowMol dependency edits are part of this takeover.

## Retained failed attempts

- 45568618: progress callback incompatible with disabled bar; fixed.
- 45569977: OOM from diagnostic force graphs and dynamic batching; fixed.
- 45570869: torch 2.2 mmap string-path requirement; fixed.
- 45574453: adaptive evaluation budget exceeded; initial script lost the first
  successful estimate. Subsequent code persists every solve or failure.
- 45575823: cancelled by this takeover after identifying boundary
  self-conditioning; partial results and stop reason retained.
- Several submissions rejected by the user's aggregate SLURM submit limit;
  these have null job IDs in `jobs.jsonl` and consumed no GPU runtime.
- `local_density_mature_v1`: launcher absent from source commit; fixed preflight.
- `displacement_development_rho01_v1`: submission saw an incompletely extracted
  source snapshot; no job created. Submission now publishes snapshots atomically.

## Required before submission claims

1. Numerical convergence of density and samples, then full-neighbourhood and
   stricter/adaptive confirmation; never remove failing parents after the fact.
2. Matched FM-only, squared, product, common, shuffled and zero-label controls,
   multiple training seeds and equal data/sample/compute budgets.
3. Independent potential evaluation and robust chemistry, diversity and mode
   coverage metrics. Strain alone is insufficient.
4. Preserved metadata and genuinely independent evaluation data, with a clearly
   specified conditional physical target and electronic state.
5. Demonstrated value against current energy-guided flow methods. The present
   code correctness and toy mechanism are foundations, not acceptance evidence.

Official deadlines: September 18 (abstract), September 25 (paper), 23:59 AoE.
https://iclr.cc/Conferences/2027/AuthorGuidelines . The existing PDF is a
reviewable audit/development draft, not a submission-ready positive-result paper.
