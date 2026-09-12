# Qualified fresh-parent result and frozen transfer continuation

The72-parent primary test is COMPLETE, with full stochastic replay and28225
independent joint-proposal MH-ratio checks. Both algorithms spend55884 raw
physical calls per replica, including the learned model's9660 one-time training
calls and the shared9360 preparation calls. The learned-minus-site mean
restrained-potential difference is-0.17350627eV, with95 percent parent-bootstrap
interval[-0.25954793,-0.09940139]eV. Replica means are-0.19438323 and-0.15262931eV.
Among45 initially non-reference parents, learned first hits are44/45 and45/45,
versus site30/45 and33/45. These are diagnostic reference-graph hits, not new
chemical discoveries. All230544 new physical calls from preparation and the full
recorded trajectories remain in the ledger.

This establishes a scoped finite-cost potential benefit at matched raw-call
budget for one composition. It does not establish universal superiority,
matched-wall-time speedup, native learned-generator superiority, quantum
chemical validity or stationary Boltzmann sampling. The source supports only
72/8192 attempts. The128-call secondary matched-cost interval crosses zero;
small-budget failures and older full-cost physical controls are retained.

The working paper now includes this result in the abstract, main evidence and
appendix, with a standalone figure. The build passes with8 main pages:
`runs/verification/fresh_reuse_completed_20260912/main.pdf`.
`research/evidence/fresh_reuse_completed_v1.json` records the quantitative result
and immutable hashes. New primary-source review adds Timewarp's acceptance
refinement, L2HMC's jump objective and TITO's transferable flow-matching dynamics
to the prior-art boundary. Generic accepted-movement training is not new.

## Six additional compositions

`transfer_development_panel_v1.json` fixes three hash-selected compositions in
each size stratum8--12 and13--24, excluding the previous8 compositions. These
are neutral singlet compositions containing C, H and a halogen, with at least
two possible non-halogen anchors; selection uses no energies or generated
geometries, and reference energies are omitted. Reserved outcomes remain unused.

Sources46137946 and audit46138020 COMPLETE. All3072 new source attempts are
retained. Supported counts out of512 are215,59,108,146,34,178; all supported
parents have eligible terminal exchanges and no validator errors were observed.
The six sizes are12,12,9,20,22,23 atoms. No source energy calls were made.

The physical test fixes the first16 supported parents per composition,96 total,
under the same frozen vector checkpoints and physical target. Preparation46139228
and full replay46139311 COMPLETE, using11946 raw calls. Training cost is charged
once across all six compositions. For learned512-call endpoints, site receives
614 calls for the first30 parents and612 for the remaining66, so each method's
primary total is70758 calls per replica, including actual preparation.
The primary uncertainty resamples composition clusters and then parents within
clusters, after averaging algorithm replicas within parent. Two targeted tests
check that repeated parents and algorithm seeds do not create false replication.

Leaf protocols `transfer_reuse_case_00.json` through05 are hashed in
`transfer_reuse_protocol_v1.json`. Input adapters under `runs/transfer_reuse_inputs_v1`
link to original immutable source artifacts and project the completed source
audit into the existing loader interface. The loader verifies the original
parent-audit hash and every projected support/provenance field; the projection
does not invent additional verification. Numerical kernels are reused unchanged.

## Live continuation

Evaluation46139699 is RUNNING, one GPU task per replica and sequential methods/
compositions within each task, respecting the two-job gpu_test submission limit.
Outputs: `runs/transfer_reuse_eval_v1/condition_XX/{site,learned_vector}_sY`.
Dependent full audit46139781 is queued at `runs/transfer_reuse_audit_v1`.
The dependent summary job is recorded in `research/jobs.jsonl` and writes
`runs/transfer_reuse_summary_v1/results.json`.

Refresh the exact live handles before acting. Do not rerun on an observation
timeout, overwrite completed traces, silently omit a failed composition, or fit
to these evaluation states. Read all six per-composition outcomes and the
prespecified aggregate before interpreting secondary results. If any required
endpoint is unavailable, the summary withholds the aggregate comparison.
Regardless of the outcome, preserve source failures and separate development
transfer from final reserved evaluation. Remaining ICLR work includes competitive
learned-generator controls, claim-appropriate independent physical/distribution
validation and an anonymous reproducibility package. Submission readiness remains
false and the full ICLR goal remains active.
