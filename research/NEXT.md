# Active raw-quality and independent-generator studies

Read `research/RAW_QUALITY_STATE_20260915.json`. Connectivity job46541711_0/1
trains replay/local/tree controls and evaluates all six frozen variants on the
reused12 development compositions. New conditional EDM adaptation is implemented
and CPU-validated; submit its frozen two-seed launcher once GPU submission slots
are free. Capped relaxation is a separate CPU diagnostic on COPIES of prior raw
outputs; original inference still has no optimizer. Track its job in the state.
All new claims await completed audits. No new architecture sweep or global-path
sampler was started. The completed checkpoint below remains unchanged.

# LatentSpring: completed fresh confirmation; raw generation is the priority

Read `research/FRESH_PHYSICS_STATE_20260915.json` and
`notes/fresh_physics_results_20260915.md`. The manuscript is
`paper/tree_working.tex` / `paper/tree_working.pdf`, titled
**LatentSpring: Physics-Informed Molecular Flow Matching from Atomic Composition**.
It compiles cleanly: 9 main pages, 16 total. Scientific submission readiness is false.

Jobs46531691_0/1 (generation/eSEN) and46531808_0/1 (GFN2) are complete.
No new BGFM study was submitted after these jobs; unrelated user jobs may be live.
Re-query specific jobs before recovery. Latest registry: v12.

## Completed result

Frozen24 new compositions, two existing model continuations, four models and32
samples per composition give6,144 new raw outputs. Old official candidates and
composition matches in both verified processed corpora are excluded. No fitting,
coefficient changes or model selection occurred on these new outcomes.

Raw graph support: Gaussian44.53%, harmonic52.73%, force update53.45%, complete-work
update52.67%. Harmonic minus Gaussian is+8.20pp, conditional paired95[5.08,11.33].
Force update minus harmonic is+0.72pp[-0.98,2.34], not an established extra validity
gain. Both force-update continuations improve energy and force under both eSEN
and independent GFN2. Common-graph energy changes are-0.01308 eV/atom under eSEN
and-0.01168 under GFN2; both paired intervals exclude zero. The prespecified
cross-potential gate passes. Source-only physics is heterogeneous across seeds.
Extra complete-work advantage over force update remains unestablished.

GFN2 has4 generated failures/6144, all graph-rejected. All graph-supported outputs
and96 reference calculations succeed. Total new costs:12384 eSEN queries and6240
GFN2 attempts. Both are approximate evaluators, not DFT or equilibrium certification.
Same two trained models are reused; this is new-composition evidence, not additional
independent model replication.

## User correction: directly valid output with little relaxation

All reported coordinates are raw generator outputs. xTB uses `--grad`, never
`--opt`. All input coordinates and atom symbols match the frozen generated samples.
No per-output energy ranking, force correction or geometry optimization occurs
at inference. Physical teacher displacements and updates occur on separate FIT
samples during learning. Preserve this distinction in every claim.

Current force-update failures/1536:474 disconnected only,3 overlap only,7 both,
231 geometry-supported but graph-rejected;821 graph-supported. Disconnection
therefore affects481/1536 (31.32%), the largest failure category. Do not interpret
graph rejection as a uniquely diagnosed chemical defect. The existing raw assay
is unchanged. Lower force does not establish few relaxation steps or stable minima.

Next essential work: one targeted connectivity intervention developed on FIT-only
data, consulting failed tree-context/product-coordinate/edge-feedback results.
Preserve all frozen harmonic and physics checkpoints. Raw generation is the primary
metric. A separate matched capped-relaxation assay can measure steps, coordinate
change, graph retention, failures and total cost before claiming little optimization.
Keep original coordinates and raw validity visible. Do not tune on fresh24 and
then describe it as untouched confirmation. A fair independent-generator comparison
and broader coverage remain substantive gaps.

The proposed additional global stochastic-path/Jarzynski reconstruction is deferred
following the user's raw-quality clarification. No such new sampler was implemented
or submitted. No architecture or coefficient sweep is queued.

Jarzynski remains the already implemented and audited LOCAL work teacher and
comparison. No global Boltzmann output law, final output density, equilibrium isomer
weights or output ESS is established. Established random-tree theory, self-conditioning,
energy-weighted FM, escorted work and task arithmetic must be credited. The scoped
molecular source/learning construction has useful evidence; ICLR acceptance or
sufficient novelty is not guaranteed. Current gains are not a matched comparison
against the original BGFM loss model.

Registryv12:84160 evaluation NN outputs+3584 FIT=87744 total;44256 source/teacher
eSEN rows plus6240 fresh GFN2 attempts, counted separately. Earlier local proposals,
projected coordinates and physical campaigns retain their separate ledgers.
Reserved722 outcomes remain unqueried; all evaluation data remain outside fitting.
Keep OMol25, max_atoms200, bond loss zero, original electronic conditions, runtime
patch architecture, NaN guards, two environments and no home writes. The user
handles authors and submission.
