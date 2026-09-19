Immediate next checkpoints:47221006/47225156 finish validation-selected GAGA
comparison;47226653 repeats the promising native endpoint pilot with GFN2.
Both have frozen protocols and all-control reporting. Publication is verified
in publication_sync_v5.json; no need to republish unchanged paper assets.
Do not turn the single-seed closed-set result into held-out or basin-calibration
claims. The next scientific missing measurement is graph/symmetry-aware basin
mass, then broader TRAIN teacher coverage, not more unclaimed rotor variants.

Current status (2026-09-19): read `research/SUGGESTIONS_STATE_20260919.json` and
`notes/suggestions_review_20260919.md` first. All original repetitions and corrected
matched-physics runs are COMPLETE. Five source runs average+3.89pp at3000 updates
(4/5 positive); all3 new runs improve at6000. The fixed-strength EGNN physical
update harms both families even with GAGA's noise schedule correctly frozen.
A new validation-only strength calibration with equal choices for FM/GAGA runs
as47221006; confirmation uses16 previously ungenerated compositions.

The supplied weighted-endpoint branch passes27 tests in the REAL DGL/FlowMol
environment. Native5.90M pilot47220840 is complete: raw graph yields283/384 base,
334/384 proposal minima,325/384 complete-work minima. This is a3-composition
closed-set integration pilot, NOT a GAGA win, full-dimensional equilibrium result,
or work-specific neural improvement. Do not add it to the main paper as such.
The manuscript adopts the reviewed prose, includes all5 source runs and the
completed GAGA follow-ups, and uses actual-coordinate PyMOL figures. Read the
latest build/sync receipt for publication status; older chronological notes below
are historical, not live job status.

Latest matched-physics qualification: the fixed-strength EGNN FM adaptation
has a VALID negative result (graph validity and joint quality collapse); do not
explain it away as the GAGA implementation bug. GAGA v1 also accidentally unfroze
its predefined gamma.gamma schedule, so that half is not the intended fixed-
schedule comparison. The adapter now preserves original trainability flags, and
a regression test proves the schedule stays unchanged through an optimizer step.
Correction job46674037 retrains ONLY GAGA from cached teachers and exact
old target choices; audit46674039 follows. No learning rate, coefficient,
or FM result is changed. See `research/MATCHED_PHYSICAL_STATE_20260915.json` and
`research/evidence/matched_physical_v1_qualification.json`. The main FlowMol
four-cell result remains its own completed study. Wider transfer is unproven;
validation-based strength control would require a separate protocol and test.

Current continuation: the raw GAGA feedback challenge is COMPLETE. Distance
self-conditioning has15.625% versus13.7695% GAGA validity on its32-composition
panel, but the composition95 interval for the difference includes zero. Global
tree feedback loses and is not adopted. Read `research/GAGA_FEEDBACK_STATE_20260915.json`.
The equal-pass physical comparison is now running as46669949, with dependent
audit46670954. See `research/MATCHED_PHYSICAL_STATE_20260915.json` and
`notes/matched_physical_comparison_20260915.md`. Source repetitions46656786 remain
separate. The published paper still contains the previously verified factorial;
no new GAGA superiority or physical-adaptation outcome is claimed yet.

Live execution update: source-repetition GPU job46656786 is RUNNING; seed2 has
completed its1000-step readout and resumed fitting. Seeds3/4 follow in the same
allocation; audit46654303 waits for completion. GAGA seed1 is complete and seed0
continues in46645569; dispatcher46636422 will launch their frozen confirmation.
See the review-response and GAGA state JSON files for the authoritative records.

Publication update: the reviewer-directed revision and completed source/physics
factorial are pushed and remotely verified. GitHub release f32e843; Overleaf
f3d2a79. See publication_sync_v4.json and publication_build_v4.json in
research/evidence. Current PDF:9 scientific main pages,26 total,25 citations.
The extra repetitions and GAGA studies remain active as described below.

Latest review follow-up: the source-by-physical-update factorial is COMPLETE and
independently audited. See `research/evidence/source_physical_factorial_audit_v1.json`.
At GFN2 force RMS<=5 eV/A, all-attempt joint yields are35.68% Gaussian,
40.625% Gaussian+physics,44.53% harmonic,47.59% harmonic+physics. Physical
training helps both sources in both continuations. The pooled source difference
with physics is+6.97pp, but continuation-specific joint-yield signs differ.
Three extra source continuations and budget curves remain in progress; read
`research/REVIEW_RESPONSE_STATE_20260915.json` for live scheduling. Normal GPU
array46643602 was cancelled while unstarted; dispatcher46654304 submits
the unchanged three repeats to gpu_test after GAGA seed1 finishes. The actual
repeat GPU job is recorded in `runs/source_replication_v1/gpu_test_submission.json`.
The matched GAGA comparison with physical adaptation is still an open follow-up.

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
