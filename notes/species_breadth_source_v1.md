# Frozen broad-development source protocol

Runtime amendment(v2): the first all-eight prefix screen failed its1e-7-A
replay gate before any oracle queries. A matched first-condition diagnostic
measured1.430511e-6 A maximum difference(RMS2.056732e-7), with unchanged input
and CPU/CUDA RNG states. Enabling torch.use_deterministic_algorithms(True) and
CUBLAS_WORKSPACE_CONFIG=:4096:8 produced exact prefix replay. This supports a
GPU arithmetic explanation for the checked prefix; it is not a global numerical
certificate. Preserve both diagnostics and all eight original failures.
The producer now uses frozen species_breadth_source_protocol_v2.json and requires
the same replay gate for every condition; the gate was not relaxed. Full
production still requires the new complete engineering screen. Primary runtime
reference: https://docs.pytorch.org/docs/stable/notes/randomness.html .

This extends the species-adapter investigation beyond condition5846 without
selecting new molecules from method outcomes. Use every row of the existing
research/evidence/development_panel_v1.json in manifest order. Reserved722
conditions remain untouched. The source differs from the specially trained
N8 work generator, so these are new experiments, not pooled replicas of it.

Freeze runs/electronic_fm_global_v1/last.ckpt and its original model-input
temperature feature. Load every trained parameter strictly and never reset
temperature weights. Sample the explicitly defined64-step midpoint displacement
map at T=1, from its original unaligned COM Gaussian prior. Add independent
isotropic COM-Gaussian terminal noise of standard deviation .025 A. This gives
a positive source density on the declared COM space even without invertibility
of the finite midpoint map. It does not make the unknown source density or its
normalizer evaluable. Relative-KL adapter training only needs the frozen source
law; absolute likelihood and path importance weights will not be reported for
this source. The .025-A noise is a declared source choice, not a physical
temperature or a claim of optimal sampling.

Physical refinement target: kT=.025851999786435 eV (300 K), same conservative
eSEN checkpoint, U=E_eSEN+.05 sum_i||x_i||^2. The source's recorded model input
kT=1 eV is retained solely as a pretrained neural feature. It is distinct from
the physical300-K target. Charge/spin/atomic identities come only from the
manifest; no reference coordinates are loaded for generation or initialization.

Initial engineering screen:32 training-source and32 separate evaluation-source
rows per condition, batch16, base seed9181, eight conditions,512 oracle queries.
Streams use distinct seeds determined by condition/stream/batch indices; record
all of them. Test finite positions/energy/force, COM, strict checkpoint loading,
and the noise-augmented sampler's reproducibility before releasing full source
production. This screen cannot qualify distribution coverage or train a paper
model. Retain every condition-level failure and do not replace failed rows.

After engineering qualification, produce4096 training and512 development rows
per condition using base seed9182 and batch64. Neither stream is used for
independent final confirmation. The producer retains separate condition records under a bounded allocation,
keeping the full eight-condition denominator. The full-panel launcher uses one
regular-GPU allocation capped at4 hours; it is released only after8/8 engineering
qualification, with all condition records retained on failure. Costs:
4608 endpoint oracle queries per condition;128 FM field calls per geometry,
with training/screen/confirmation costs additionally accounted.

The next matched comparison will use the already fixed1000-step nonlinear and
typed-linear recipes, batch16 and two independent training streams. It requires
a generic trainer that validates these new source metadata and does not invent
the old path work. That trainer is not yet implemented. Every condition and
failure must be retained. Outcome-based architecture tuning makes this a
development panel; independent confirmation and strong coupling-flow/HMC
comparisons are still needed. No claims follow from source-production success.
