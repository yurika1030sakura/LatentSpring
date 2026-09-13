# Current research status — September 13, 2026 UTC

Authoritative checkpoint: `research/MOBILITY_RELAXATION_RESULT_20260913.json`.
The full ICLR goal is active and scientifically unachieved. The completed matched
training and larger-cohort checks establish no useful screening advantage.

## Paired mobility diagnostic

All four producer cases and audits are complete: 32 FIT parents, 128 arms,
5,564 raw calls (256 shared initial/repeat, 1,638 roots, 3,670 collective).
The 8,448-call cap is respected. Full replay and independent physical/constraint/
Armijo checks cover all 2,654 queried optimization trials; a separate compact-
result reconstruction checks all pair contrasts, means, costs and stop counts.

Initial mean destination-source gap is 1.30174 eV; best bounded root-only and
collective gaps are 0.62051 and 0.54324 eV. Their paired difference is -0.07727 eV,
with descriptive within-composition parent interval [-0.22250, 0.08523].
Collective mobility lowers destinations an additional 1.10282 eV, but sources
also improve by 1.02555 eV. This does not establish a specific exchange benefit.

Root-only arms: 59 converged, four budget exhausted, one minimum-step stop.
Collective arms: five converged, 49 budget exhausted, ten minimum-step stops.
Only 2/32 pairs have all four arms converged. Equal caps do not mean equal actual
cost or optimization accuracy. Neither intrinsic chemical minimum-energy gaps
nor the usefulness of collective geometry is resolved. This is optimization,
not a sampler, equilibrium result or AI contribution.

Jobs46212367 and46212970 are terminal. Summary:
`runs/mobility_relaxation_summary_v1/results.json`. Read
`notes/mobility_relaxation_decision_v1.md` for the next bounded diagnostic.

## Matched gate pretraining

Direct utility800 is compared with gate-target500 plus utility300 for both
linear/neural architectures and two seeds. Identical sampled-index streams and
an Adam reset at500 are verified across recipes. All final/control metrics,
12,776 supported-pair cases, and24 phase-boundary/final head gradients pass the
declared mixed absolute/relative checks. Maximum independent gate discrepancy
is5.40e-14; maximum FD discrepancy is4.01e-6 across differently scaled objectives.

Original12-parent internal rates (1e-3 eV per expected whole-prefix raw call):

| Recipe/model | Seed0 | Seed1 |
| --- | --- | --- |
| Direct linear | 4.4046 | 4.2739 |
| Direct neural | 4.3856 | 4.4093 |
| Pretrained linear | 4.2358 | 4.2121 |
| Pretrained neural | 4.3426 | 4.3412 |

No screen is4.1675; fixed physical is4.7331. Pretrained-minus-direct neural
intervals are below zero for both seeds. Pretraining and longer matched training
do not resolve the competitive deficit. Summary:
`runs/gate_distillation_summary_v1/results.json`.

## Larger reused cohort

The frozen evaluation-only cohort has48 parents, six compositions and96 physical
first128-call prefixes. Neither those parents nor the original12 selection
parents enter gate fitting. Two composition identities were absent from fitting.
These trajectories previously served scalar-model evaluation, so this is reused
INTERNAL evidence, not a final test. The corrected condition5 control is used.
All1,620 attempts,1,582 scored pairs and6,144 used force states retain energy,
force, charge/spin and raw-query provenance. No new physical calls are used.

Six independent audits replay every model/control metric and30,058 model-pair
cases; maximum discrepancy2.79e-13. Aggregation weights six compositions equally,
then parents and two trajectories within parent. The all-composition no-screen
rate is0.009051543 eV/call; physical screening gives0.008617472. Pretrained neural
rates are0.008851281 /0.008725512, point changes of+2.71% /+1.25% versus physical
screening but-2.21% /-3.60% versus no screening. Both comparisons' intervals span
zero. Neither neural model establishes benefit over linear or matched thinning.
The two unseen compositions also show no consistent neural gain. No tested
screen establishes improvement over no screening in this cohort.

These are fixed-source expected substitutions in recorded prefixes, not actual
screened chains, saved queries, wall-time gains or equilibrium accuracy. All
intervals fix compositions, resample parents with both trajectories and are
not multiplicity-adjusted. Summary: `runs/screen_transfer_summary_v1/results.json`.
The larger-cohort ranking does not support raising confidence in ICLR readiness.

## Decision and next work

Resolve optimization convergence and saved constraint stops before building a
new learned transport. A continuation would require frozen caps and cached-prefix
replay for all budget-exhausted arms; no continuation is submitted yet. Keep
minimum-step failures and converged controls. Do not change the physical target
or confuse uncontrolled relaxation with a valid MH proposal.

Screening-only tuning remains stopped. The earlier gate/transfer round used
zero new physical calls; the mobility diagnostic adds 5,564. All older negative
evidence and costs remain. The 722 reserved outcomes and every evaluated cohort
stay out of fitting. The paper remains scientifically unready. Prior status:
`notes/archive/status_through_gate_transfer_20260913.md`.

Verified development PDF: `runs/verification/mobility_relaxation_20260913/main.pdf`
(9 main pages,25 total). Build evidence:
`research/evidence/mobility_relaxation_build_20260913.json`. Scientific
submission readiness remains false.
