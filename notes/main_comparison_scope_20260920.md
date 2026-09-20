# Main comparison agreed with the user, 2026-09-20

The main comparison subsection contains four methods: Gaussian FM, EDM, GAGA,
and our self-conditioned FM with harmonic-mixture source plus learned physical
head. It excludes the optional hydrogen readout and baseline-plus-our-head rows.
The main metrics are graph validity and all-attempt joint graph-valid/GFN2-force
RMS<=5 eV/A yield. The user explicitly declines adding new full-schedule sampling
runs; do not schedule extra1001-call EDM or651-call GAGA for this table.

The common conditions are the64-composition panel,16 draws/composition/fit,
20,000 OMol25 training rows, shared2,381,566-parameter EGNN backbone, and
960,000 backbone training-example forwards. All methods use128 effective backbone
calls/output. Our self-conditioned FM uses15,000 two-pass updates, while the
other parents use30,000 one-pass updates at batch32. These are matched backbone
forward budgets, not matched total FLOPs, optimizer steps, or wall time.

Our method additionally receives1024 eSEN force queries/fit,20,000 updates to a
7106-parameter head, and64 head calls/output. This extra supervision, training,
and inference must be disclosed. Baselines are not required to adopt our modules.
Some archived zero-strength controls execute unused physical-head calls for
sampler verification; any baseline efficiency timing must explicitly account for
or remove that diagnostic overhead. Do not infer equal runtime from backbone calls.

Current Gaussian FM/EDM summaries use2 archived fits, while GAGA and the final
self-conditioned FM use5 fits; repeat counts and uncertainty must be explicit.
The one-pass source/head factorial is a separate two-fit diagnostic and must not
be labeled a four-cell ablation of the exact final self-conditioned model.
