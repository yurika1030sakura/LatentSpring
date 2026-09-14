# ICLR readiness checkpoint

The research goal is unachieved. The manuscript is an internal development draft.
It is not scientifically submission ready and no acceptance probability is assigned.

What is currently demonstrated:

- A normalized bond-free spatial tree-source implementation, checked source laws,
  original electronic metadata and a reference-qualified monomer benchmark.
- Repeated geometric improvements from structured sources in prior experiments,
  but no established repeated learned-source chemical-validity increment.
- A properly parameterized geometry self-conditioning baseline improves graph
  support by5.47 percentage points over the one-pass baseline, with paired95
  interval[3.06,7.94] and composition interval[4.10,6.77] on12 reused conditions.
  Both training seeds improve. This is a known-technique baseline result.

What does not establish a new useful AI contribution:

- Coordinate-NLL source heads, fixed source-tree context, source-utility learning,
  current-geometry tree attention, fixed-tree product-coordinate flow and localized
  relation feedback all fail their declared extra-benefit gates.
- The product-coordinate flow guarantees contact connectivity, yet chemical graph
  validity becomes substantially worse. A geometric guarantee alone is insufficient.
- Relation-head weights learn from coordinates, but localized codes do not improve
  consistently over geometry-only or pooled feedback. An active gradient is not
  evidence of useful molecular generation.
- None of these results establishes a calibrated Boltzmann energy distribution.

Latest completed test:

`research/DUAL_GEOMETRY_STATE_20260914.json` records a FAILED internal-geometry
gate. Original/geometry-SC/fixed-geometry/time-geometry graph counts are
384/410/167/328 and327/369/156/315 out of768. The time-scaled variant loses to the
original representation. Geometry SC again improves on original with new draws
from the same two trained models; this is a draw-stream confirmation, not new
independent model training. The pooled improvement is4.43 percentage points,
with paired95 interval[2.02,6.90] and descriptive composition95 interval[2.73,5.92].
See `research/evidence/geometry_sc_fresh_stream_v1.json`.

Next most useful bounded work:

1. Audit the original native endpoint generator before further architectural
   additions. The original30,000-step checkpoint restores and its hash verifies;
   its native conditional sampling has not been measured by these displacement
   adaptations. A successful restoration is not a performance result.
2. Compare Gaussian/moment controls and structured sources under the stronger,
   correctly parameterized geometry-SC backbone. Fixed prior design can itself
   be an AI-method contribution; a new trained module is not universally required.
   Its originality and usefulness still need support against relevant prior art.
3. Freeze any chosen method before a new-composition confirmation. Do not tune
   envelopes, priors or filters on this repeatedly used12-condition panel.

No new experiment is currently queued. All reserved outcomes remain unqueried;
all evaluated coordinates/results stay outside fitting. The user handles authors
and submission. The ICLR goal remains unachieved. No energy-law claim is supported.
