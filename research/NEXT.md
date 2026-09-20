# Current checkpoint —2026-09-19: shared-backbone correction confirmed

The shared-EGNN physical-head study is COMPLETE and independently audited.
Read research/evidence/matched_connection_audit_v1.json and
research/evidence/matched_connection_selection_v1.json. On16 fresh compositions
(512 attempts per model), FM joint graph-valid/GFN2-force<=5 yield improves
8.01->15.04%, +7.03pp [2.73,11.91], with both fits improving. GAGA given the
same7106-parameter architecture,128 TRAIN compositions,1024 force queries per
parent, and20000 head updates improves10.94->17.58%, +6.64pp [3.71,9.77].
Corrected FM-minus-GAGA is-2.54pp [-6.64,1.76]: no established ordering.
This demonstrates correction transfer to independently trained flow and
diffusion backbones; do not turn it into an unsupported matched GAGA win.

All four parent/head fits use identical physical supervision budgets. FM has64
head calls and GAGA128, in addition to128 backbone calls each. The native GAGA
schedule/observation noise and every parent tensor remain frozen. Protocol
matched_connection_v1 is fixed before sampling; eight separate validation
compositions select strength1 for each algorithm before any fresh test output.
All4096 validation/test outputs received successful fixed-coordinate GFN2.
The19 relevant tests pass. No optimization or energy queries occur in sampling.

Array47317335 (teacher/fit/validation) and47317610 (fresh test) are complete.
State: research/MATCHED_CONNECTION_STATE_20260919.json. Immutable experiment
source: fd055b8 in runs/matched_connection_v1/source. Registry v30 counts5120
new neural trajectories (1024 TRAIN and4096 evaluation),4096 eSEN queries,
4096 GFN2 readouts and80000 head updates. No BGFM job is currently running.

The paper now foregrounds the learned physical correction and reports both
matched generators. Complete-system FlowMol results below remain valid and
separate. Publication build v9 is verified:9 main pages/35 total,27 cited references,
49 standalone files. Canonical and standalone PDF text match. Publication
sync v9 verifies GitHub429155f and Overleaf114b275, including all49 exported
files and the GitHub PDF. The existing HOLY Overleaf mirror is fast-forwarded.

# Previous complete-system checkpoint —2026-09-19

All new studies are COMPLETE and audited. Read
research/evidence/published_model_confirmation_audit_v1.json,
research/evidence/context_confirmation_audit_v1.json, and
research/evidence/common_model_diagnostics_v1.json.

On one common16-composition panel, joint graph-valid/GFN2-force<=5 yield is
38.87% for the frozen parent,46.88% for the8178-parameter20k-update force head,
46.09% for a37540-parameter wide head,46.68% for the37586-parameter contextual
head,9.77% for GAGA128 and12.11% for GAGA651. The preselected contextual-head
comparison passes: +7.81pp [4.88,10.94] over parent. Extra neighborhood complexity
has no established generation benefit. All three long fits improve the parent.
The paper's existing paired-update model is38.09%, or40.04% without terminal
noise. A declared later same-data2k/20k ablation gives+3.71pp [1.76,5.86].

These are COMPLETE-SYSTEM comparisons with different backbone sizes, histories,
training data and physical supervision. The older SHARED-EGNN GAGA comparison
still has uncertain ordering; do not call the entire36-point gap a new-head or
matched-algorithm gain. Main generator conditions remain neutral organics.

Model recipe: configs/research/latentspring_force_correction_v1.json.
Run with python -m scripts.generate_latentspring in the flowmol environment;
its checkpoint/config/condition arguments match the underlying audited generator.
No energy calls or optimization occur during generation. No global Boltzmann
or minimum-energy guarantee is established. Jarzynski remains scoped to the
local-work theory and controlled molecular experiments.

Publication build v8 adds the model/controls, common-panel GAGA results, and a
readable result figure. It compiles to9 main scientific pages/33 total, with27
cited references and46 standalone files. Publication sync v8 verifies GitHub53d28f6 and Overleaf0b701d3; all46
export files and the GitHub PDF hash match the build. Registry v29 includes completed costs, distinguishing1024 new
supplemental neural trajectories from512 extra noise-derived output records.
No BGFM experiment is currently queued. The two manifest failures occurred
before any generation and are preserved in PUBLISHED_MODEL_CONFIRMATION_STATE.

Never write to home or push the home origin remote. Keep the two Torch
environments separate. User wants Chinese replies, essential work, useful
improvements, clear input/output definitions and scientifically faithful figures.
For figure work use .agents/skills/scientific-figure-design/SKILL.md.
Historical statuses below do not override this checkpoint.

## Next actions

- Build v9 is published and verified in publication_sync_v9.json.
- The shared-EGNN physical-head transfer is now complete, including an equally
  supervised GAGA control. Do not rerun the same test or select new settings on
  its outcomes. The supported contribution is correction transfer, with no
  reliable FM-over-corrected-GAGA ordering.
- Keep the effective small head and the original complete-system comparison.
  The broader contextual head did not improve generation beyond the small head.
- For the deadline, prioritize a coherent submission around harmonic source and
  learned physical correction. User handles author details and OpenReview.
