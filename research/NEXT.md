# Next — completed main-generator pairing comparison

Read `research/ORBIT_PAIRING_STATE_20260913.json` and
`notes/orbit_pairing_fm_v1.md`. The ICLR goal remains active and unachieved.

The experiment is complete, with every one of2560 generated structures included.
Counts passing the stated graph check out of512: warm47, independent continuation65,
rotation61, collision-aware62, typed assignment+rotation74. The collision-aware
candidate does not establish an increment beyond continuation or standard matching.
Versus typed matching its paired fraction difference is-0.02344, interval
[-0.04297,-0.00391]. This is a fixed-model/development comparison, not a training-
replicated superiority result. Do not turn its positive warm-only comparison or
lower path overlap into an AI novelty claim.

What is now available:
- `research/evidence/orbit_pairing_five_method_v1.json` and `.csv`: all five models,
  source checksums, matched3000 training rows, all structural outcomes replayed,
  per-condition results, validator exceptions, costs and paired uncertainty.
- `research/evidence/generator_reference_registry_v1.json`: exact checkpoints and
  inference settings for all models. Retain independent and typed-matching
  continuations as conventional development controls. Their selection used this
  development panel; they have not been qualified on reserved outcomes.
- `research/figures/orbit_pairing_v1/orbit_pairing_v2.pdf`: support categories and
  paired uncertainty. Graph perception is not an electronic validity certificate.
- `runs/orbit_pairing_train_v3`, `runs/orbit_pairing_eval_v2`,
  `runs/orbit_pairing_audit_v1`, `runs/orbit_pairing_typed_v1`: full source artifacts.
- `runs/orbit_pairing_energy_v1/energy/results.json`: the prescribed structural
  gate failed, so energy scoring was skipped with zero new molecular oracle calls.
- `research/evidence/orbit_pairing_scheduler_v1.json`: jobs46323226,46323424,
  46324210,46325717 and46327728 all completed0:0. Query Slurm before assuming later
  work is absent. Earlier cancelled/rejected submissions remain preserved.

The next research decision:
1. Close this collision-aware rotation-search recipe to scale-up. Do not vary
   its angle candidates, penalty weight or training duration to seek a win on
   the observed outputs. Earlier routing and latent-mass toy sweeps stay closed.
2. Work from the stronger conventional generator controls and identify a concrete
   remaining failure mechanism for a new learned-distribution contribution.
   Pairing-cost reduction alone did not establish that contribution. A new design
   must have a plausible advantage beyond standard type/rotation matching and a
   bounded direct molecular test. No replacement mechanism is declared solved.
3. Keep the current scope honest: graph support/observed structural diversity are
   not calibrated energy distributions. The ordinary FM data target is empirical
   OMol25, not automatically a Gibbs ensemble. A useful generation claim does
   not require perfect mixing on every case, but a thermal claim needs actual
   compatible evidence. No independent new physical law is mandatory.
4. If a new hypothesis survives, freeze its core comparison before querying new
   evaluation outputs. Use fresh disjoint streams; retain the complete original
   results, all failure categories and preparation/inference cost. Do not use
   the reserved outcomes or this panel's generated geometries as fitting data.

Implementation facts to preserve:
- Optional orbit pairing has a Gaussian source after shared Haar augmentation.
  Typed matching also randomizes within identical conditioning-label groups and
  requires permutation-invariant edge features. Shape is unchanged under this
  physical symmetry; arbitrary target atom-row order is not preserved.
- The independent-Gaussian velocity-to-score identity does not apply to correlated
  endpoint pairings. This pilot is FM-only and leaves the original BGFM hooks
  intact. Do not silently enable the old force proxy on the new coupling.
- The midpoint64 T1 displacement sampler plus0.025-A COM noise has no qualified
  absolute likelihood. Do not score it with a mismatched clamped q0.95 or claim
  importance ESS. Keep original charge/spin and model-input temperature.

Protected:722 reserved outcomes; old12/18 evaluated cohorts; the2560 new evaluated
outputs; separate FlowMol/oracle environments; no home writes; OMol25 primary,
bond loss zero, max_atoms200. User handles authorship/submission. The existing
`paper/angular_working.tex` is still a diagnostic draft, not a new ICLR manuscript.
