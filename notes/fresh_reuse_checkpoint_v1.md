# Fresh-parent cost validation checkpoint

The previous turn made concrete progress by completing the EACF proposal audit,
repairing the fresh-source startup failure and updating the paper. This turn
finishes the fresh source and its structural audit, implements and tests a
per-parent finite-query experiment, completes physical preparation and full
replay, and submits the frozen paired evaluation plus dependent audits/figures.
The ICLR goal remains active and unachieved.

## Completed evidence

- Source46134723 and audit46134750 COMPLETE:8192 fresh independent random draws,
  72 structurally supported parents (0.87890625%), four perceived connectivities,
  zero validator exceptions. The unsupported8120 attempts remain explicit.
  No source energy labels or density/importance weights were computed.
- Source parent identities:30 `FC[SH](F)(F)(F)F`,27 `CS(F)(F)(F)(F)F`,
  13 `FC(F)[SH2](F)(F)F`,2 `F[SH3](F)C(F)(F)F`.
- Preparation46135442 and full replay46135607 COMPLETE:all72 parents,
  64 fixed local warm steps,9360 acknowledged and requested raw physical calls.
  No fresh state is used for training or selected by energy.
- Four targeted tests pass. They check separate parent budgets, billing of
  evaluated-but-rejected proposals, censored parents, unavailable endpoints,
  and averaging algorithm seeds before parent-cluster bootstrap.

## Frozen scientific comparison

`research/evidence/fresh_reuse_protocol_v1.json` fixes unchanged target, both
trained vector checkpoints, two algorithm seeds, all72 eligible parents,
512-call learned and1024-call physical paths, and a1024-microstep ceiling.
`fresh_reuse_analysis_plan_v1.json` is frozen before evaluation submission:
the PRIMARY comparison is the entire cohort with512 learned inference calls
per parent and exactly matched total physical calls including9660 training calls.
For72 parents, site receives648 calls for the first6 and646 for the other66.
The shared130-call warm cost per parent is included on both sides. Other budgets
and cohort prefixes are descriptive secondary readouts; do not replace a failed
primary result with a favorable selected secondary result.

The source is still only one development composition. Structural support does
not imply quantum chemical validity. Query-count stopping assesses finite-cost
output and need not preserve a stationary distribution. Neither first hits nor
lower potential establish a Boltzmann law. Independently evaluate the learned
component on other compositions before claiming generalization. Preserve the
strong physical controls and the limits of repurposed EACF comparisons.

## Running pipeline and exact continuation

The initial four-task GPU submission was rejected withQOSMaxSubmitJobPerUserLimit,
before assigning a job or making any query. Preserve the rejectedv1 directory.
Recovery runs two tasks, one replica per task, with site and learned methods
sequentially under the same frozen protocol.

- Evaluation46135971 is RUNNING at this checkpoint, output
  `runs/fresh_reuse_eval_v2/{site,learned_vector}_s{0,1}`. Do not resubmit on an
  observation timeout. Completed chunk artifacts are immutable.
- Full replay and independent density-ratio audit46135988 is PENDING afterok
  evaluation, output `runs/fresh_reuse_audit_v1`.
- Summary/figure46136261 is PENDING afterok audit, output
  `runs/fresh_reuse_summary_v1/results.json` and `figures/fresh_reuse.pdf`/`.png`.

Re-query those exact handles. Inspect any failure before recovery; preserve
its failed trace, oracle counts and completed chunks. On full success, read
the declared `primary_comparison` first, inspect the figure, then update the
paper from this prospective checkpoint to the actual result. If the primary
comparison fails, do not scale the frozen guide merely because an old four-parent
first-hit result was favorable. Diagnose whether geometry, reverse proposal
probabilities or learning objective limits its incremental value.

Current working build: `runs/verification/fresh_reuse_protocol_20260912/main.pdf`,
8 main pages. Formal and empirical method reconstruction is ongoing; formatting
checks are not scientific submission readiness.
