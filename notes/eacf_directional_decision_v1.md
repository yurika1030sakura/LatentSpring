# Audited EACF proposal pilot and next evidence decision

The EACF direction-augmented MH baseline is implemented, tested on known targets,
qualified against real forward/inverse interfaces, and evaluated in all four
frozen capacity/seed arms. Sampling job46134040 and all four audit46134148 tasks
completed. The new experiment used6518 raw physical calls; replay and aggregation
used none. Exact checkpoint, report and audit hashes are in
`research/evidence/eacf_directional_comparison_v1.json`.

The normalized learned vector reaches the diagnostic reference graph from all
four common starts in both256-step pilots. The physical site control reaches3/4
and2/4; neither EACF capacity reaches it in either seed. All1024 attempted EACF
flow moves are rejected;823 have supported endpoints. Independent MH-ratio and
full producer/real-flow replay pass. Full57D Jacobian checks on one seed per
capacity agree to2.92e-13 or better.

This positive comparison has a strict interpretation: the EACF checkpoints were
trained for refinement, not MH acceptance from warmed states. It is not a native
EACF/FAB result or evidence that its architecture cannot work. Median potential
penalties dominate in these runs; the auxiliary/volume correction also matters.
An acceptance-trained learned-flow control or an appropriate native generator
comparison remains relevant if making a general learned-sampler superiority claim.
Do not present this poor repurposed baseline as the decisive ICLR evidence.

The learned model still costs12446/12456 raw calls including preparation versus
site2734/2756. That is a real adverse four-parent stress test. The correct next
question is measured reuse over fresh parents and independent compositions,
with training charged once, generation attempts retained, and inference costs
measured. No hypothetical amortization or equilibrium claim is justified.

The fresh geometry-only source v1 failed before sampling because DGLGraph lacks
`.cuda()`. Its snapshot/log and cancelled dependent audit are retained in
`geometry_only_source_failure_v1.json`. Recovery uses `.to('cuda')`, the same
frozen8192-sample protocol and distinct v2 outputs. Jobs46134723/46134750 are the
producer and dependent audit; refresh terminal state before using their artifacts.
The first batch has now been produced. There are no energy labels or weights.

Next: finish and audit that source, freeze a paired fresh-parent sampling/reuse
protocol before querying energies, then evaluate learned-vector and physical
site controls. Include all unsupported attempts; fresh seeds for condition0 are
not new molecular compositions. Do not expand the mixture network or open the
reserved722 outcomes on the strength of the present pilot.

Assessment: more defensible than the inherited loss/density narrative and worth
pursuing, but not submission ready or high-confidence acceptance. The remaining
critical evidence is the value and scope of learning, not a new physical law or
a perfect generator. The candidate mechanism is learned geometric proposals for
reversible graph/coordinate exchanges; novelty must be distinguished from prior
learned MH and directional-mixture methods with a concrete ablation and benefit.
