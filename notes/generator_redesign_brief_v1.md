Current follow-up: the specified generator-calibration candidate is now implemented and audited. Read `notes/latent_mass_calibration_v1.md` and `research/LATENT_MASS_STATE_20260913.json`. It supplies a constructed-target coupling signal, not established distinctive neural or molecular-generator superiority. Earlier open-design text below is historical.

# Next bounded design task: learn the generator's distribution

The completed routing experiments do not establish the AI contribution needed
for the paper. Cooperative panels lose to strong single-edit controls; the
single-edit learner has only a small unresolved increment over its exact
zero-learning ablation. Further scorer capacity or longer routing sweeps are
not the next priority.

The next candidate to assess acts on the generator's relative density, rather
than on another post-generation edit preference. The question is whether valid
nonlocal energy comparisons can improve physical probability allocation that
local, per-parent constraints fail to determine. This is a design hypothesis,
not an implemented replacement or a diagnosed cause of the preceding failure.

There is an existing starting point: `cfm_mol/clamped_density.py` defines a
deterministic, composition-clamped COM position ODE, with matching
`sample_clamped_flow` and `log_density_clamped_flow` functions. Its default
`q_0.95` is explicitly a separate sampling law. It is NOT the likelihood of the
archived CTMC endpoint, history-dependent self-conditioning, steric retractions,
or the production T1-plus-noise stream. Preserve this distinction and consult
the existing density/adjoint evidence before repeating numerical audits.

The first bounded feasibility task is:

1. Read the actual BGFM loss/conditioning definitions and identify exactly what
   is held fixed across samples: atomic composition, original charge/spin,
   per-atom charge tags and any edge features. Source and transported endpoint
   densities must use the SAME conditioning. A change of chemical identity must
   not silently change the model's conditioning or its normalizer.
2. Define one controlled generator whose sampler and density refer to the same
   deterministic field and terminal time. Inspect existing solver-convergence
   and gradient evidence; close only missing checks. Any modification of the
   velocity, clipping or support conditioning changes that law and its density.
3. Formulate the candidate learning target for valid paired points. If samples
   are conditioned on a common validity set D, the unknown probability of D
   cancels in a within-condition density ratio. This alone does not ensure high
   validity, global coverage or a novel objective. A complete-graph sum of squared
   pair differences may merely equal an existing grouped variance loss; do not
   rename that algebraic equivalence as an AI contribution.
4. State the distinction from the current BGFM energy term, energy-based flow
   training, ratio/score matching and the primary sources in
   `notes/ai_novelty_boundary_20260913.md`. In particular, mode blindness and its
   remedies already have prior art, including DiffCLF. Positive Gaussian noise
   and the data-flow-matching objective prevent a blanket disconnected-support
   argument. No molecular mode-weight reference has yet diagnosed this failure.
5. Only if a distinct and useful mechanism survives that comparison, implement
   one small controlled distribution test with known probabilities, followed by
   a bounded real molecular test. Measure generated distributions and valid
   sample retention as well as inference/training costs; loss reduction is not
   enough. Retain energy-based and zero-learning controls from the outset.

No new expensive physical data is needed to assess this specification. The
current finite-work catalogues, paired geometries and physical baselines remain
available. No evaluated12/18-parent state or722 reserved outcome may enter new
fitting. Keep OMol25 primary, bond supervision zero, max_atoms200 and the two
environments separate. The flow-matching architecture remains the project basis.

This file deliberately does not assert that the candidate is original or will
work. The next deliverable is one concrete, prior-art-aware main-method
specification and its minimal test, not another broad family of scorer variants.
