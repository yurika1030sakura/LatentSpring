Current priority: follow the supplied manuscript review. Read
`research/REVIEW_RESPONSE_STATE_20260915.json` and
`notes/review_response_20260915.md` before older status entries.
The main text now foregrounds source structure, replay subtraction, the same-norm
control and absolute physical quality. Work/rotor theory remains in the appendices.
Job46645569 runs the missing Gaussian physical-update factorial cell for both
continuations, then resumes checkpointed GAGA seed0. GAGA seed1 is46635650_1;
confirmation dispatcher46636422 has been redirected to the live dependencies.
Source repetition array46643602 adds three new continuation seeds and budget
curves. None of these pending results may be claimed as established in the paper.

Current experiment status (2026-09-15): GAGA feedback training is active.
Read `research/GAGA_FEEDBACK_STATE_20260915.json` and
`notes/gaga_feedback_challenge_20260915.md` first. Jobs `46634916_0` and
`46635650_1` train two feedback models per seed, then evaluate validation baselines.
Job `46636422` waits for both and dispatches validation selection, fresh-panel
confirmation and a raw-output audit. The eventual confirmation GPU job will be
recorded in `runs/gaga_feedback_v1/confirmation_dispatched.json`.
The original seed1 startup failure occurred before training; it was a seed-check
bug, fixed while preserving the frozen baseline seed41401. No outcome or protocol
was changed. No new generator improvement is established yet. The published
manuscript remains the verified molecular-figure release below.

Current checkpoint (2026-09-15): the editorial and molecular-figure revision is
published and remotely verified. The paper explicitly defines flow inputs/outputs, uses actual
molecular coordinates and includes supplementary flow/rotor animations. Its
canonical build has 9 scientific main pages, 25 total pages and 24 cited references.
The new standalone build and publication receipt are recorded in
`research/evidence/publication_build_v3.json` and
`research/evidence/publication_sync_v3.json`. GitHub release: `398a825`; Overleaf: `239932c`. Preserve Overleaf collaborator commit `0eecd2802fcd235e54e0e3f3ca50e7b3db44cf15`.

The user has also authorized an attempt to improve on GAGA. The new experimental
`cfm_mol/connectivity_feedback.py` has three passing mathematical/compatibility tests,
but no trained-result claim. Next: freeze a validation-only selection protocol,
compare geometry and auxiliary-tree feedback at accounted training/inference
budgets, and evaluate the selected setting on fresh held-out compositions.
Do not revise the paper's generator ranking before results support it.

Historical checkpoints follow; the paragraph above takes precedence for current
publication status. Full experiment records are in
`research/STRONGER_EVIDENCE_STATE_20260915.json`.

Active follow-up: read `research/STRONGER_EVIDENCE_STATE_20260915.json`.
All matched-quality, curvature-teacher and paired-student experiments are audited.
The curvature teacher improves ESS; its extra neural benefit did not pass the
two-potential gate. Complete the user's expanded editorial/figure review and
rewrite, incorporating only supported claims. Record actual flow input/output
and preserve the collaborator's latest Overleaf history during publication.

Latest publication checkpoint: all expanded generator and component experiments are complete
and audited. The manuscript now defines its symbols and method roles explicitly,
includes a reader's guide and workflow figure, and reports direct component
benefits. Scientific main text:9 pages; current build evidence is
`research/evidence/publication_build_v2.json`. Read
`research/BENCHMARK_EXPANSION_STATE_20260915.json` and
`notes/reader_and_component_review_20260915.md`. The independent EGNN source
contrast is positive, while EDM/GAGA have higher overall validity in that setting.
No universal generator-superiority or global Boltzmann claim follows.

Current continuation: the user authorized stronger generator comparisons and broader evaluation.
Read `research/BENCHMARK_EXPANSION_STATE_20260915.json` and
`notes/benchmark_expansion_20260915.md` before the last publication checkpoint below.

# LatentSpring: focused submission draft assembled

Read `research/RAW_QUALITY_STATE_20260915.json`, `paper/REPRODUCIBILITY.md`,
`notes/paper_framing_20260915.md`, and `notes/rotor_work_results_20260915.md`.
Canonical PDF: `paper/tree_working.pdf`; source: `paper/tree_working.tex`.
Title: **LatentSpring: Physics-Informed Molecular Flow Matching from Atomic Composition**.
The publication build has8 pages of scientific main text; the AI-use statement
ends on page9, with14 pages total. Plain-text title/abstract is
`paper/submission_abstract.txt`. Build hashes and standalone-package checks are in
`research/evidence/publication_build_v1.json`. Default paper links and the
Overleaf package now use LatentSpring. Publication details are recorded in
`notes/publication_sync_20260915.md`.

## Current publication scope and user priorities

The user explicitly corrected the writing direction: a paper is not an experiment
diary. Focus on supported contributions; discarded alternatives need not appear
when no claims depend on them. The final draft removes unrelated failed architecture
branches and the capped-relaxation experiment while retaining their research records.
Keep the relevant controls, actual raw rates, training histories and scoped limits.
Jarzynski stays as qualified local-work theory and a positive molecular mechanism
experiment. Do not claim global3D Boltzmann generation, fast convergence to minima,
a new fluctuation law or work-specific neural superiority.

No full OMol25 run is required for the current scoped paper. Source comparisons
use an existing pretrained backbone plus3000 selected OMol25 structures per
continuation. Never call this full-corpus training. A larger-subset study could
strengthen future evidence but is not a prerequisite that every discarded branch
must meet before this manuscript can be reviewed.

## Evidence in the draft

- Harmonic source improves raw graph support over Gaussian by5.86pp on12 development
  compositions,8.83pp on10 additional, and8.20pp on24 fresh compositions. Matched
  covariance-Gaussian controls also lose on the earlier panels. The same two fitted
  model continuations are reused; intervals are conditional on models/compositions.
- Fresh24 raw support: Gaussian44.53%, harmonic52.73%, force update53.45%, work
  update52.67%. The force update lowers energy/force in both continuations under
  eSEN and independent GFN2. All reported generator outputs are unoptimized.
- Independently adapted EDM was strengthened to30000 total updates on the same
  3000 FIT rows: raw graph13.22% at128 calls and12.96% at1001 calls. Its backbone,
  pretraining history, capacity and compute differ. This is a conditional adaptation
  comparison, not native published EDM/SOTA performance or a pure source ablation.
- Controlled Jarzynski experiment:3 FIT methyl rotors,512 GFN2 energies each,
  3 fixed escorts and128 repetitions at8/32/128/512/2048 particles. Complete work
  passes all normalizer checks and beats both incomplete-weight controls at2048
  draws in all6 nonidentity cases. Mean TV0.0636 versus0.1532 energy-only (58.5%
  lower) and0.1134 without Jacobian. Free-energy RMSE0.158–0.734 meV. These are
  constrained, interpolated-potential targets; they do not validate global generator
  equilibrium or an extra neural-training advantage.

The main AI contribution candidate is the specific latent-connectivity harmonic
source and its integration with molecular FM and physical learning. Prior tree
mathematics, self-conditioning, escorted work and parameter arithmetic are credited.
This is a credible scoped submission story, not a certificate of ICLR-level novelty
or acceptance. The ultimate ICLR goal remains unachieved; the manuscript is ready
for review. Do not turn acceptance uncertainty into a requirement for a perfect
generator or an endless sequence of unclaimed experiments.

## Completed research outside the chosen method

Endpoint-connectivity regularization did not pass its gate and was not adopted.
The capped-relaxation diagnostic did not establish short relaxation:0/384 generated
samples and1/24 references converged within20 steps. Median generated RMSD was
about0.39 Angstrom. These are retained in `notes/raw_quality_results_20260915.md`
and their audited artifacts; no relaxed copy replaces original generation.

## Execution and provenance

All relevant jobs are terminal:46541711 (connectivity),46542830 (relaxation),
46543959 (initial EDM),46546006 (rotor work),46546237 (extended EDM).
Job46543268 never trained and was cancelled while pending after unsuccessful
in-place partition updates. Its exact-source replacement and all records remain.
Re-query specific jobs if resuming; unrelated user jobs can still be active.
No additional BGFM experiment is queued.

Latest registry isv13:94912 evaluation NN outputs+3584 FIT=98496 total records.
Source/teacher eSEN rows remain44256. This follow-up adds9679 GFN2 attempts,
including8142 relaxation,1 smoke and1536 rotor-grid calls; this branch's GFN2
confirmation/follow-up count is15919. Older physical campaigns remain separate.
Rotational numerical Monte Carlo draws are not NN outputs or oracle queries.
Reserved722 outcomes remain unqueried and all evaluation data remain outside fitting.

Preserve OMol25, max_atoms200, bond-loss weight zero, original electronic states,
runtime patches, NaN guards, the two environments and the no-home-write rule.
Do not silently promote correlated-source FM into an independent-Gaussian score
identity. The user handles authors and submission. Preserve the frozen main models.
