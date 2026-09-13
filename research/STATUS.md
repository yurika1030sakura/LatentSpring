# Current research status — September 13, 2026 UTC

Authoritative checkpoint: `research/GATE_DISTILLATION_RESULT_20260913.json`.
The full ICLR goal is active and scientifically unachieved. The completed matched
training and larger-cohort checks establish no useful screening advantage.

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

Stop tuning/scaling the tested screening-only recipes. Inspect existing
whole-COM escorted paths before another collective proposal: v1/v2 accepted
only4/64 paths per replica even when the graph guide repaired endpoint support.
A new diagnostic must separate intrinsic chemical energy gaps, geometry limits
and path/proposal-density costs. Matched root-only versus collective relaxation
of BOTH source and destination can be informative, but only as teacher feasibility
with explicit convergence/support/failure accounting. No new protocol or physical
job is frozen/submitted. See `notes/post_screening_reconstruction_20260913.md`.

Jobs46203103,46203225,46206100 and46206282 are complete. This round has zero new
physical calls; prior data construction and numerical repair costs remain.
All older scalar, geometry, action, strong-control and transfer negatives remain.
The722 reserved outcomes and every evaluated cohort stay out of fitting. The
paper remains scientifically unready. Prior status:
`notes/archive/status_through_source_force_screen_20260913.md`.

Verified development PDF: `runs/verification/gate_distillation_transfer_20260913/main.pdf`
(9 main pages,24 total). Build evidence:
`research/evidence/gate_distillation_transfer_build_20260913.json`. Scientific
submission readiness remains false.
