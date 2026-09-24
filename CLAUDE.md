# Active checkpoint —2026-09-24: paired geometry-recovery pilot running

An independent architecture pipeline, job48239292, tests the7,489-parameter
GeometryMomentContext against an equal-parameter radial-feature control on the
same development panel. Both start from the original frozen parents and reuse
the physical heads; only the new context encoder trains. Source db45585 is in
runs/geometry_context_recovery_v1/source. Its768 baseline outputs are symlinked
from the recovery pilot and must not be counted twice. Five module tests pass,
including gradients through a frozen real EGNN. See geometry_context_recovery_v1.json,
geometry_context_cost_accounting_v1.json, and GEOMETRY_CONTEXT_STATE_20260924.json.
Sampling adds128 small-context calls to128 backbone and64 physical-head calls.
The contexts study is independent of which continuation candidate wins. Both
pipelines retain their immutable source snapshots. Main-paper results are unchanged.

User authorized substantive improvement of raw chemical validity. Job48231033
is the single-GPU pipeline for runs/geometry_recovery_v1. Re-query its live
state; research/GEOMETRY_RECOVERY_STATE_20260924.json records submission and
subsequent status. The initial six-task array was rejected by gpu_test's submit
limit, so the same scientific experiment runs sequentially within one job.
No other project job was changed. Training uses immutable source a57c0c2 in
runs/geometry_recovery_v1/source; launcher6765bd5 was copied by Slurm.

Read research/evidence/geometry_recovery_v1.json and
notes/geometry_recovery_20260924.md. Fits0/1 each receive4000 matched continuation
updates for replay, perturbed-endpoint recovery, and recovery with local
distance/angle supervision. Same20k references, batches, original architecture,
source, and sampler; existing force heads stay frozen at strength4. Four
targeted tests pass, including actual two-pass EGNN gradients. No bond labels
or test-time geometry optimization are introduced. This pilot changes backbone
training; it is not yet the proposed additional geometry-network architecture.

The24-composition development panel (12 per size bin,16 draws each) is separate
from the primary64. It is not represented as historically untouched. Frozen
and continued controls are both included;3072 attempted outputs are scored at
fixed coordinates. Candidate advancement checks raw geometry, geometry-plus-force,
both fits, composition intervals, and graph-valid distinct-connectivity yield.
The pipeline invokes audit_geometry_recovery automatically at completion.
No new improvement result is available yet; manuscript v28 stays unchanged.

# Previous checkpoint —2026-09-24: round2 revision and chemical geometry audit

The following checkpoint supersedes the dated records below. The combined-design
diffusion job 47380645 and illustration job 48184024 are COMPLETE. No experiment
is pending. The diffusion adaptation did not improve either target: EDM joint
8.15 -> 0.00%, GAGA 9.47 -> 0.05% on the paired two-fit subsets. Frozen-head-only
transfer remains positive; do not conflate these experiments or claim both
designs transfer successfully. Read joint_design_transfer_audit_v1.json and the
updated research/JOINT_DESIGN_TRANSFER_STATE_20260920.json.

Round2 accepts Overleaf coauthor commit 2bfca98: Molecular Generation title,
formal introductory prose, and a filled three-item contribution list. The
coordinate operation is centering, not scale normalization; the FM network
predicts velocity. Preserve these scientific corrections when merging prose.
See notes/ROUND2_RESPONSE_20260924_ZH.md for every adopted/adapted suggestion.

Main figures are now research/figures/round2_method_v1 and round2_results_v4.
Use scripts/research/compose_round2_method.py and render_round2_results.py;
historical section-rendering scripts may overwrite the new completed results.
The PyMOL renderer now preserves aromatic bond orders and checks its round trip.
The method illustration uses raw C9H9NO2 coordinates and a 2D inferred graph.
It comes from a separate, fully recorded 128-output validation illustration
panel, not the primary 64-composition benchmark. No geometry was optimized.
The older gallery and generation animation remain archived but are superseded
in the current paper. Three of four old gallery structures fail geometry checks.

All 20,480 main FM/GAGA parent/corrected outputs were reassessed with the declared
PoseBusters geometry subset plus zero assigned radicals. This is not the full
PoseBusters validity suite. Geometry-plus-force yield is FM 1.64 -> 4.43% and
GAGA 2.40 -> 6.45%, versus the original graph-plus-force metric's FM 7.21 -> 24.00%.
These definitions must remain distinct. Connectivity and planarity are material
remaining weaknesses; presentation improvements do not establish solved chemistry.
The 512 validation references yield 470 geometry passes. Code, per-output records,
reference records, and all denominators are in round2_* evidence and
cfm_mol/chemical_geometry_review.py. All existing primary benchmark rates are unchanged.

Registry v39 accounts for 4,096 completed diffusion-adaptation outputs, 128 new
illustration outputs, all GFN2 attempts, and 120,000 diffusion parent updates.
No new physical-head fitting or eSEN queries. Final paper build and synchronization
are recorded in publication_build_v28.json and publication_sync_v28.json.
GitHub paper commit 016295c and Overleaf commit 40f0b9b are synchronized; the
HOLY Overleaf mirror is also current. Local canonical and standalone builds
pass at 9 main pages, 47 total, 33 references. Cloud compilation and OpenReview
submission were not performed. The manuscript remains a submission draft.

# Historical checkpoint —2026-09-20: both-design diffusion transfer running

User requests original-baseline versus both-design transfer for EDM and GAGA,
without additional source-only/head-only evaluation arms. Job47380645 is RUNNING
on four GPUs from immutable source3286abd in runs/joint_design_transfer_v1/source.
Read research/evidence/joint_design_transfer_v1.json,
research/JOINT_DESIGN_TRANSFER_STATE_20260920.json, and
notes/joint_design_transfer_20260920.md. Eight new tests passed; all four fits
passed initialization/batch-schedule checks and the first250 training steps.

This experiment retrains four diffusion parents (two fits each) from the paired
original random weights for30000 one-pass updates of batch32 on the same20000
OMol25 rows. Harmonic tree covariance is used in noising and every reverse
innovation, and conditions the denoiser through the existing invariant edge
channel. The source tree is an auxiliary noise variable, never a chemical bond
label. EDM starts from the tree mixture; truncated GAGA uses its Gaussian data
variance approximation plus the harmonic corruption covariance. The source law
and scalar VP schedule are fixed; no claim of optimal new GAGA truncation.

The two existing FM physical heads transfer unchanged at strength4. Thus this
is design transfer with parent retraining, not zero-shot transfer of the entire
model. No new physical-head fitting or eSEN queries. Only the combined arm is
newly generated:4096 outputs/GFN2 attempts on the existing64-composition panel.
Original EDM controls reuse the previous two-fit archive; GAGA controls use
fits0/1 of the five-fit archive, not its five-fit mean. Compare only compatible
fit subsets. No H flow or long-schedule inference. Every result is retained.
The job automatically audits and summarizes into runs/joint_design_transfer_v1/audit.json.
No new scientific performance result is available yet. Paper build v16 merges
the previous repetition and direct-head-transfer subsections into one fourth
experimental subsection. A new original-versus-both table contains four explicit
pending cells; two-fit baseline values are already filled. Existing repetition,
weight-transfer, and optional H results plus the figure remain in the main text.
The first three experimental subsections and their results are unchanged.
Paper build v19 replaces the main comparison, ablation, and merged-transfer
tables with three vector figure groups in research/figures/experimental_story_v4.
All per-fit data remain visible; pending outcomes receive no invented marks.
The title is now LatentSpring: Harmonic Sources and Physical Corrections for
Molecular Flow Matching. Abstract/introduction/discussion follow the two-design
story. The main methods now describe EGNN velocity output, detached distance
feedback, and the actual midpoint sampler without terminal noise. Earlier
FlowMol-specific details remain documented separately. Structured-diffusion
covariance conditioning and parent retraining are explicit. See
notes/narrative_experiment_alignment_20260920.md and experimental_graphics_review_v1.json.
Build v19:9 main pages,46 total,32 references; local canonical and standalone
text match. Publication synchronization is recorded in publication_sync_v19.json.

# Current checkpoint —2026-09-20: three additional baseline transfers complete

Job47372153 completed successfully. All12288 additional outputs and fixed-coordinate
GFN2 attempts are audited; no training or teacher calls were added. Read
research/evidence/other_baseline_transfer_audit_v1.json and
research/evidence/source_head_factorial_v1.json. All three transfer primary contrasts
have positive gains in both archived fits and positive three-target Bonferroni
crossed-fit/composition intervals. Joint yields: EDM8.15->22.22%, Gaussian FM
6.25->19.92%, one-pass harmonic FM7.23->24.27%. These all use the same original
FM-trained heads, strength4, without retraining or H readout. Three GFN2 numerical
failures remain in denominators. All generation checks and physical log audits pass.
Registry v38 includes every new attempt and48 verification-only replay outputs.

The four-cell FM analysis is a post-hoc interpretation of this frozen transfer
experiment: Gaussian6.25, harmonic7.23, Gaussian+head19.92, harmonic+head24.27% joint.
Both-vs-physical-only gain4.35pp, nominal crossed95CI[1.56,7.13]; source-only
joint gain0.98pp CI[-0.68,2.54] is uncertain. The two parents are one-pass EGNNs
without self-conditioning and receive the same frozen head from the previously
trained self-conditioned FM. This is NOT a four-cell ablation of the exact final
self-conditioned backbone with separately trained source-specific heads. Both-fit
EDM comparison is separate from the five-fit GAGA confirmation. No universal
or cross-architecture claim. Paper build v14 reorganizes Experiments into setup, a four-method/two-metric
main table, and a four-cell source/head ablation. Main text contains result
comparisons; detailed training/sampling settings are in Appendix B. The one-pass
ablation is explicitly distinguished from the self-conditioned main generator.
Canonical and standalone builds match:8 main pages,45 total,31 references.
PDF pages6--7 were visually checked; tables stay with their subsections.
Publication synchronization is recorded separately in publication_sync_v14.json.

# Current checkpoint —2026-09-20: five-fit physical correction and weight transfer confirmed

ALL scientific jobs are COMPLETE. No further training or sampling is pending.
Read seed_replication_audit_v2.json (energies after uniform SCC completion),
seed_replication_audit_v1.json (original failures), and cross_generator_head_audit_v1.json.
On64 new compositions (20 with17--28 atoms and44 with29--40),5120 attempts per
model across5 independent fit pairs give FM joint7.21->24.00% and GAGA8.75->25.37%
with the learned physical head. The three NEW fits confirm FM+15.625pp,
composition CI[12.34,19.01], crossed fit/composition CI[8.50,22.43], with every
fit improving. GAGA's three-new-fit gain is16.89pp. One FM initialization is weak;
retain it. With H flow FM26.33% vsGAGA25.74%; neither joint nor graph superiority
is stable across fitted models. Do not claim comprehensive GAGA dominance.

Verbatim FM-head->GAGA transfer improves new-fit joint yield16.54pp
[13.02,20.08], crossed[12.66,20.67]; all3 gains positive. Reverse transfer gives
17.12pp [13.61,20.77], also positive in all3. No head retraining or target-parent
force labels train the transferred head. Architecture/vocabulary are shared;
do not extrapolate this to every architecture or chemistry.

H readout vs fixed radial rule lowers new-fit all-output energy19.31meV/atom
[15.17,23.67]; crossed reduction interval[11.62,28.05]. Six initially unconverged
GFN2 calculations (all graph-invalid) converge with1000 instead of250 SCC
iterations, otherwise identical conditions and coordinates. No geometry
optimization. Original failures retained; graph and joint arrays unchanged.
The original first-pass H gate was incomplete; the completion audit is explicit.

Registry v37:281152 evaluation parent trajectories,8960 FIT trajectories,
290112 total parent-generation records,203706 eSEN rows,110631 GFN2 attempts.
Derived evaluation readouts sincev34:28800. This campaign adds285000 optimizer
updates. Inference-package verification repeats are separate from scientific
attempts; all failed diagnostics are retained. Reserved722 outcomes unqueried.

Model API: cfm_mol.latentspring_generator.LatentSpringGenerator;
CLI: python -s -m scripts.generate_from_bundle. The10-model/5-fit bundle is in
/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/releases/latentspring_egnn_20260920.
Cross-head selection uses head_family; only the small source head is loaded.
CPU/CUDA replays passed. CUDA scatter-add is not bitwise deterministic in the
archived sampler; deterministic API/direct-sampler checks match exactly, and
archived coordinates agree within3.31e-5A with identical graph/geometry labels.
The package includes all models, not a selected best initialization.

Paper title: LatentSpring: Transferable Physical Corrections for Molecular
Generation from Atomic Composition. The source and endpoint update are central;
paired weight updates/Jarzynski remain in appendices. FLEXDOCK, ConfDiff and
Lai2026 are cited: no first-endpoint/first-force-network claim. Build v12 verifies
9 main pages,42 total,31 references,60 exported files. Publication sync v12
verifies GitHubd338a05 and Overleafd844db0 plus the HOLY mirror.
All60 source files and the GitHub PDF match the build; coauthor history retained.
51200 per-output scores and a NumPy-only reproduction script are public in GitHub.
The full10-model weight bundle is stored locally in HOLY releases, not uploaded
to GitHub. No OpenReview submission has been performed.

# Previous running checkpoint —2026-09-20: independent training replication

User requests further work toward a strong ICLR paper. Current jobs47340136
and47340267 run the fixed shared-EGNN comparison on64 new compositions across
five fit pairs. Fits2/3/4 are new independent parent/head/H-model training;
fits0/1 reuse archived weights. Read SEED_REPLICATION_STATE_20260920.json and
seed_replication_v1.json. No new scientific claims are qualified yet.
The metadata-only panel amendment uses20 smaller and44 larger compositions,
with exact zero overlap in both processed corpora and no generator-based selection.
Read seed_replication_panel_resolution_v1.json. All settings are fixed; preserve
all new fits and failed runs. Do not select a favorable seed or change strength.
The old-fit64-composition runs and audits are complete; three new parent pairs
are still training. Cross-head transfer is separately frozen in
cross_generator_head_v1.json: old fits audited, new-fit job47342296 waits on
47340136. Read CROSS_GENERATOR_HEAD_STATE_20260920.json. Prior-art review added
FLEXDOCK, ConfDiff and Lai2026; no first-force-network or first-endpoint claim.
The inference-only model bundle passes CPU and GPU replays for FM/GAGA.
Published v11 below remains the completed evidence; subsequent text/figure edits
must not claim results from these running jobs.

# Current checkpoint —2026-09-20: conditional hydrogen flow physically confirmed

All scientific studies below are COMPLETE. No training or scientific evaluation
is still running. The CLI integration replay is complete and is verification only.
Read HYDROGEN_FLOW_STATE_20260920.json and hydrogen_physical_confirmation_audit_v1.json.
On30 fresh compositions (960 attempts/setting), the shared-EGNN FM model plus
conditional H flow gives graph27.40%, joint26.04%; its unchanged strength4 parent
is25.31%/24.17%. Against the fixed radial rule, the learned readout reduces
all-output energy15.21meV/atom [10.10,21.71], with similar graph/joint yield.
Moving all H instead of only detached H reduces energy another10.25meV/atom
[6.96,14.72] with identical validity. The same H decoder also improves GAGA.
Equally augmented FM-minus-GAGA graph is+3.33pp [0.52,6.15]; joint is+2.29pp
[-0.63,5.21]. No comprehensive or joint-quality superiority claim.

Original hydrogen yield-only development screen remains failed. The separate
physical hypothesis was frozen before the fresh30 test. Hydrogen decoration is
established prior work (Quetzal); the contribution is the controlled conditional
readout and demonstrated energy benefit. Do not claim global Boltzmann sampling.

Recipe: configs/research/latentspring_hydrogen_flow_v1.json; entry point:
python -m scripts.generate_hydrogen_flow. See notes/hydrogen_flow_model_20260920.md.
This is an OPTIONAL shared-EGNN readout; the main FlowMol model has not been
tested with it. CPU replay reproduces saved results to4.77e-7 Angstrom.
The atom-normalization/mobility2x2 candidate fails its independent confirmation
and is NOT promoted; preserve all variants, outcomes and240000 fitting steps.
Registries v32--v35 include every new study: cumulative250432 evaluation parent
trajectories,7424 FIT trajectories,197562 eSEN rows and77703 GFN2 attempts.
8320 derived evaluation readouts and1024 TRAIN radial probes are separate from
parent trajectories. H fits add640000 training forwards and512 diagnostic forwards.

Paper v11 adds the H method, all controls, fresh results and an actual-coordinate
PyMOL figure. Build verified:9 main pages,38 total,28 references,52 exported files.
Twelve targeted tests and CPU/CUDA CLI replays pass. Publication sync v11
verifies GitHubf8e7ce5 and Overleaf91bc8f6 plus the HOLY mirror.
All52 exported files and the GitHub PDF match the build; coauthor ancestor
1b64efd is retained. This is a published draft, not an OpenReview submission.

# Previous checkpoint —2026-09-20: calibrated correction complete

Strength4 gives shared-EGNN FM graph22.17%, joint20.80%, versus equally
calibrated GAGA21.00% and20.02% on32 fresh compositions. FM improves its own
strength1 joint yield by5.66pp [2.73,9.38]. FM-minus-GAGA is only+0.78pp
[-4.00,6.54] with mixed fit ordering: no stable/comprehensive superiority.
FM sampling time is14.6% lower; force medians essentially tie, and its
all-output GFN2 energy is0.0527eV/atom higher [0.0273,0.0786].
Read connection_tradeoff_audit_v1.json and connection_tradeoff_diagnostics_v2.json.
The earlier strict graph-retention calibration failed and stays failed. The
separate joint-yield objective was declared before the fresh32-composition test.

Both experiment arrays47320976/47321520 are COMPLETE. Registry v31 adds5120
evaluation trajectories and5120 GFN2 attempts, with no new training/eSEN.
One GAGA strength1 GFN2 failure remains in the denominator.
Build v10 is verified:9 main pages/36 total,27 cited references,50 standalone
files. Publication sync v10 verifies GitHub2c2acbc and Overleaf71690d7; the
existing HOLY Overleaf mirror is updated.

The user said continue. A TRAIN-only diagnosis reuses cached native trajectories
and endpoints without new queries: training_attachment_diagnostic_v1.json.
On512 native final outputs, FM has346 connected heavy skeletons versus314 for
GAGA, but60 hydrogen-only fragmentations versus18. FM has128 H atoms with no
heavy contact versus30;119/128 are within1.5 covalent-radius sums of a heavy atom.
Next bounded candidate: atom-normalized physical messages plus balanced
per-atom force mobility, tested in a2x2 ablation with identical displacement
norms, parameters, TRAIN records, updates, and equal options for GAGA. This is
implemented and running in array47326221, source7af98e5. Read
ATOMWISE_CONNECTION_STATE_20260920.json. It is not established yet. Do not repeat failed tree-manifold or
endpoint-connectivity penalty recipes; notes/tree_manifold_brief_v1.md and
endpoint_connectivity_audit_v1.json record their failures.

# Previous checkpoint —2026-09-19: shared-backbone correction confirmed

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

# BGFM project guide


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

Historical milestones below do not replace this checkpoint.



## Active takeover branch

Current review branch: `iclr2027-arch-fix` (Claude architecture update, then
Codex audit). Read `notes/claude_update_review_v1.md` before using the optional
index-split diagnostic or interpreting its particle results. Main molecular
production remains on the original immutable snapshot.

The original48-arm campaign and all8 geometry conditions are now complete.
The active reconstruction is described in `research/RECONSTRUCTION_CURRENT.md`:
conditional proposal densities, explicit validity support and reversible
chemical/coordinate moves. These are research prototypes. The uniform chemical
move has a useful molecular diagnostic; a learned-policy or ICLR-level advantage
has not been established. `research/NEXT.md` is the current execution checkpoint.

**Claude continuation entrypoint:** read `CLAUDE_HANDOFF.md`, `research/NEXT.md` and
`research/CHEMICAL_WORK_POLICY_STATE_20260913.json` first. They record the current method,
live job handles, exact next commands and remaining scientific gates. The core
candidate is ready for continuation; the paper is not submission ready.
Historical checkpoints below and in STATUS must not replace live scheduler
verification. `research/NEXT.md` is the current action list.

The user's latest priority correction takes precedence over older next-action
lists: focus on AI framework novelty and a fast real molecular demonstration.
Do not require all-case optimization or perfect mixing first. The19-arm optimizer
recovery is paused before submission. A new edit-conditioned reversible transport
is implemented and training46215808 is complete for two seeds and three model
ablations. Read `notes/edit_conditioned_bridge_v1.md` and the current NEXT.
Its usefulness and distinctive AI contribution remain unproven; test actual
corrected proposals next. Keep essential correctness, costs and data isolation.

The user's September12 scope clarification is authoritative for this campaign:
read `research/CLAIM_AND_BENCHMARK_SCOPE.md`. Include learned-generator baselines
as well as physical controls and evaluate realistic reuse costs. Universal
perfection, success on every element and an independent new physics law are not
submission requirements. Correctness and evidence for the actual claims remain
mandatory; a four-parent all-cost stress test is not the sole readiness criterion.

The current learned-guide manuscript is `paper/angular_working.tex`, with its
own angular/joint-geometry methods, proofs and completed pilot evidence. It is an
explicit development draft. Normalized learned proposals repeat short graph first
passages, but strong physical controls, preparation cost and poor distribution
diagnostics prevent a superiority claim. Pendant-fragment maps are implemented
and their physical pilot is audited, including a preserved Al validator failure.
The direction-augmented EACF MH baseline and its full real-map replay are complete.
These are frozen refinement checkpoints reused as proposals, not native EACF/FAB
performance. The original 72-parent concentration-10 benefit and its independent
GFN2 ordering remain recorded, but the completed concentration-64/400 controls
remove the demonstrated competitive advantage. All 28 strong-control arms, their
full replay and the cost-accounted summary are complete. Frozen six-composition
transfer also has no demonstrated average gain. The stiffness student overfits;
direction contributes over 98% of its held-out-parent local-teacher KL.
Eight disjoint training compositions have now been generated/audited, with two
zero-support cases retained. The constrained geodesic proposal now has fully
audited root and joint graph/radius/two-root implementations. On real training
starts, joint support increases from690/768 to745/768; all3,072 attempts replay
and750 independent forward/reverse densities agree. This is geometric progress,
not a demonstrated learned advantage. Local force/work fits fail as unrestricted
probability teachers. On valid arcs their higher acceptance still gives less
expected one-step energy decrease than uniform arcs; do not blindly distill them.
The complete physical-chain pilot46176878 and audit46176906 are complete:
48 FIT parents,36,864 calls,4,668 independently checked joint ratios. Site arcs
improve the128-call training potential-change readout by0.07615 eV over legacy
site64, with22.4% more sampling runtime. Uniform arcs do not establish average
full-chain gain. This remains a physical baseline result.

The new scalar conditional energy module is now implemented, trained and audited.
`cfm_mol/conditional_arc_energy.py` predicts masked-context radial interaction
curves; `arc_energy` integrates its actual scalar score into the full joint
proposal. Work-only and work-plus-force objectives each have two frozen800-step
models. All four improve internally withheld nonlocal work prediction by roughly
19--24% relative to the physical site score; this is not a molecular sampling gain.
The protected438-context data and all final model/baseline metrics are replayed.
The added928 withheld endpoints use1,856 raw calls; training uses no new oracle.
All jobs are terminal:46178737/46178918 and46179382/46180176. Read
`research/CONDITIONAL_CHAIN_RESULT_20260912.json`,
`notes/conditional_nonlocal_arc_learning_v1.md` and the updated NEXT before action.

The scalar learned-chain comparison is now COMPLETE on48 internally withheld
parents and six compositions. All36 arms reach their declared caps and8,682
joint MH ratios pass independent checks. Work and work-plus-force lose the
model-data-cost comparison by+0.5453 and+0.4929 eV, respectively; both intervals
exclude zero. At equal128-call inference, differences are+0.0469 and-0.0055 eV,
with intervals crossing zero. The force-loss addition has no established gain.
Do not scale these frozen models or call their19--24% work-prediction improvement
an effective sampler. Canonical-move diagnostics also show no clear benefit.

A near-pole cancellation was repaired without relaxing the frame check. The
failed learner reuses352 cached calls. One affected physical-control suffix had
to be regenerated for consistent arithmetic, adding1,392 calls. Corrected
trajectories use66,588 calls; actual research expenditure is67,980. Original
failures and discarded suffixes remain preserved. Partial-arm timing excludes
its previous failed-attempt time, which is separately recorded; no matched-wall-
time claim follows. See `research/CONDITIONAL_CHAIN_RESULT_20260912.json`.

Bounded geometric accepted-utility learning is now implemented, trained and
audited. It differentiates observed forward/reverse joint proposal densities
while retaining no-grad production draws. Two seeds improve the internal offline
utility-per-call proxy by about 2.7%, with stable importance weights. Its physical
behavior data costs 18,110 raw calls, not the scalar model's old overhead.

The fresh actual-oracle check is COMPLETE: 72 internal source states, 3,456
attempts, 6,708 raw calls and 3,354 independently replayed MH ratios. Actual
utility differences are near zero and both parent intervals span zero. Small
effects remain possible, but no fresh-proposal or full-chain benefit is
established. Do not scale this frozen geometry-only recipe. Those preceding geometric-utility jobs
are terminal; the new comparison below is active. Re-query Slurm. Read
`research/ACCEPTED_UTILITY_STATE_20260912.json`.

The six-arm bounded CONDITIONAL action/geometry comparison is now COMPLETE.
Neither action-only nor joint establishes an internal utility gain versus
physics. Geometry-only reproduces the prior +2.7% proxy, whose fresh-proposal
benefit remains unqualified. All final/baseline metrics and576 full densities
replay; eight trained-head gradients pass finite differences. Jobs46192301 and
46192339 are terminal. Read `research/ACTION_GEOMETRY_STATE_20260912.json`.
Do not scale these frozen weights or claim useful coupling from the architecture.

The bounded delayed-screen comparison is now COMPLETE: two linear and two
neural300-step models with fixed zero/physical controls. All metrics and6,388
supported-pair cases pass independent checks. Learned internal query-rate gains
are31/43% over no screening, but the fixed physical screen has a higher93% point
gain and neural-minus-linear intervals span zero. No useful neural contribution,
actual saved calls, complete-chain or wall-time gain follows. Jobs46196124 and
46196212 are terminal. Read `research/DELAYED_SCREEN_STATE_20260912.json`.

Actual dispatch screens before the candidate oracle query and distinguishes
valid geometry from scored proposals. The zero screen reproduces previous RNG
streams, queries and states. A fake-worker test exposed buffered stdout hiding
an oracle response; the byte-buffer reader fixes it, and RPC wall time is now
recorded separately. Original physical results and old timing are retained.

The general source-force screen and whole-prefix accounting comparison are now
COMPLETE. Forward decisions use only cached source force; candidate force is
used after querying for the reverse gate. The correction includes both actual
gate probabilities. Four300-step linear/neural models and fixed physical/work/
zero controls plus FIT-matched thinning are audited. All6,388 pair cases pass
independent checks, and all whole-prefix metrics replay. Both neural seeds lose
to the fixed physical screen; only one beats thinning. No repeatable useful
neural advantage is established. Do not scale these weights. Jobs46199581 and
46199747 are terminal. Read `research/SOURCE_FORCE_SCREEN_STATE_20260912.json`.

The fixed-source whole-prefix proxy includes initial/nonjoint costs but does
not replay changed chains. Neural gains6.06/2.19% versus no screen are below the
fixed physical13.57%. The earlier93% value used conditional joint-query rates;
it is not a whole-sampler speedup. No new physical queries or actual savings
occurred. Force provenance and original36/12 split remain audited.

The matched800-step direct versus gate-pretraining comparison is now COMPLETE.
All eight models and boundary/final objectives are audited, with matched sampled
index streams. Pretraining does not resolve the original12-parent selection
failure. The larger48-parent, six-composition reused evaluation is also complete:
no screen establishes a benefit over no screening. All30,058 model/pair cases
pass independent checks. The two unseen composition identities do not establish
consistent neural benefit. No new physical queries or actual savings occurred.
Read `research/GATE_DISTILLATION_RESULT_20260913.json`.

Stop scaling/tuning these screening-only recipes. Small positive point changes
against another screen do not replace the no-screen comparison. All current
jobs46203103,46203225,46206100 and46206282 are terminal; re-query Slurm before
recovery. The larger cohort is evaluation-only and previously served scalar
experiments; it is not an untouched final benchmark and must not enter fitting.

The128-cap mobility continuation and boundary checks are COMPLETE. All53 capped
arms were continued with cached-prefix replay, adding3,768 raw calls. Root and
collective convergence are63/64 and46/64;20/32 full pairs converge. The paired gap
contrast remains inconclusive. All14 minimum-step stops were checked with frozen
raw/distance-adjusted force steps, adding94 raw calls. Adjusted directions give
12 resolved decreases and2 increases. This motivates actual-energy backtracking
and constraint-aware optimizer recovery, not a new sampler or AI novelty. Read
`research/MOBILITY_CONTINUATION_RESULT_20260913.json` and
`notes/mobility_relaxation_continuation_decision_v1.md`. All four new jobs are
terminal. The former next recovery is paused by the user priority correction above.
Preserve the128-arm record, target, graph and electronic state. Keep evaluated cohorts and722 reserved outcomes out of fitting. The full
ICLR objective remains active and scientifically unachieved.
The previous manuscript is `paper/main.tex` with
`sections/M1_refinement.tex` and `M2_refinement_proofs.tex`, describing the actual
exact-entropy refinement candidate. It is explicitly a development draft.
`paper/legacy_audit.tex` preserves the historical audit entrypoint. The build
script accepts an optional second entrypoint argument. Do not confuse either
formatting pass with scientific submission readiness.

Read `research/STATUS.md` for completed jobs, failures, current experiments and
scientific gates; `research/PLAN.md` records the deadline plan. The user has
explicitly authorised framework reconstruction. The new position-only branch
in `cfm_mol/clamped_fm.py` and `scripts/research/train_clamped_position.py`
factorises a frozen composition prior from an unaligned conditional geometry
flow. Endpoint-head experiments end at T=0.8; a separately trained residual
velocity head ends at T=1 and records `position_parameterization=displacement`
in its checkpoint. Never guess the head semantics or silently reinterpret old
weights. See `notes/factorized_geometry_flow.md` and
`notes/displacement_geometry_flow.md` for the respective targets. No molecular
energy-training advantage has yet been established.

## Historical density-audit findings

The existing experiments show improved local ordering of an archived scalar
readout. They do not establish a calibrated density or improved generation.
The archived density routine integrates FlowMol3's endpoint prediction as a
velocity and uses a frozen-trajectory gradient. It remains available for
reproduction, with an explicit warning in energy training. Its historical
name `log_density_via_flow` is not evidence of likelihood correctness.

The new `cfm_mol/clamped_density.py` defines a separate deterministic,
zero-centroid clamped positional ODE, converts the endpoint head using the
actual schedule, and differentiates through the complete midpoint solve and
prior. Its default terminal time is 0.95. This is **q_0.95**, not the joint CTMC
sampler's conditional or endpoint density. No archived molecular performance
number came from this corrected path. It needs new training and convergence
validation. `configs/audit/omol25_clamped_cnf_smoke.yaml` is a smoke template,
not a production recipe or a reported experiment.

The later endpoint-entropy prototype in `cfm_mol/endpoint_entropy.py` uses an
actual-terminal-noise proposal score, with VSD/DMD and denoising prior art. It
is a gradient prototype, not a completed molecular method. The invariant
proposal-score critic failed independent qualification and both longer-training
seeds; no molecular actor was updated. The global innovation Gaussian auxiliary
also failed its bounded screen. Do not repeat these recipes or promote their
small-panel passes. Read `research/NEXT.md` and `research/NOVELTY_DECISION.md`.

The newer `cfm_mol/stein_calibration.py` adds conservative residual score
calibration. Its fixed four-feature version reduces estimated score risk on an
independent16384-parent panel but fails unfitted radial/angular checks. Only the
frozen calibration component has been tested; it does not qualify an actor
update or establish novelty. See `notes/stein_calibrated_entropy_candidate.md`.

The current exact-entropy refinement route freezes the FlowMol FM base and
learns a separate invertible adapter. The nonlinear species-coupling model,
invariant conditioner, exact centered volume and same-context affine/typed
controls are implemented. The actual source has cross-product features and is
not guaranteed O(3)-invariant. Current production uses an inversion-mixture
source and inversion-averaged potential; read the frozen parity protocol before
training or evaluating. The force average has a MINUS sign on the inverted force.
The quadratic target coefficient is 0.05, implemented as restraint/2 with
restraint_eV_A2=0.1. The one-query raw-energy training estimator is valid for a
linear expectation, not inside importance weights or MH acceptance.

The projected N8 re-score retains small convex-minus-affine changes of
-.013366+/-.002517 and -.012610+/-.002150 nat on shared2048 parents. Path ESS
remains approximately1/2048; no calibrated sampler or established ICLR novelty
follows. Keep the base frozen for unknown-source-entropy cancellation. The affine
control's full conditioned map remains nonlinear. Full eight-condition source
generation and the generic trainer are complete; corrected48-arm production
is submitted as45914819/45914826. These new FM64-plus-noise sources have no
evaluated path weights. Read NEXT and the handoff for current qualification,
external baseline environment status and remaining experiments.

## Locked project decisions

- Flow matching, bond-free: `total_loss_weights.e = 0` on OMol25.
- OMol25 primary data, `max_atoms = 200`.
- Loss family: FM + lambda_1 force + lambda_2 grouped energy (+ lambda_3 anchor).
- Reuse the installed FlowMol3 backbone. The existing joint pipeline remains
  patch-based; the user-authorised conditional branch may replace its training
  objective without modifying the shared external installation.
- Keep `cfm_mol/projection.py` and inactive bond-dependent code for later work.
- Preserve non-finite-loss guards. Do not infer achieved Boltzmann consistency
  merely from the loss family, infinite capacity, or the potential's coverage.

## Architecture and entry point

`scripts/run_train.py` reads config, **pops `mol_fm.bgfm` before
`model_from_config`**, constructs FlowMol3, calls `patch_flowmol`, then calls
`patch_flowmol_bgfm`. FlowMol3 rejects unknown constructor keys.

`cfm_mol/flow_model.py` patches interpolation, the inner endpoint-to-velocity
helper, sampling steps and optional discrete projections. Bond-free mode
disables discrete projections. The sampling retractions do not define a
smooth invertible CNF and are excluded from the corrected clamped density.

`cfm_mol/bgfm_train_hook.py` adds physics losses; perturbation shards are loaded
outside Lightning's DataLoader. Energy-only training requires those shards,
not force labels on the main FM batch. Historical force diagnostics remain
available; `force_endpoint_to_velocity: true` fixes head conversion only,
not finite-time force-target, aligned-prior or retraction assumptions.

`energy_density_options` selects the energy implementation:

```yaml
energy_density_options:
  mode: clamped_cnf
  terminal_time: 0.95
  parameterization: endpoint
```

Absent this block, legacy mode reproduces existing configurations and warns.
Changing modes changes the objective and requires separately identified runs.

## Theory references

- `notes/bgfm_method.md`: corrected score/density/grouped-loss specification.
- `notes/group_overlap_bound.md`: finite-set spectral-gap bound; standard
  mathematics, not a claimed new theorem or a continuous basin-mass guarantee.
- `paper/sections/A1_proofs.tex`: ideal residual results with audit caveats.
- `notes/archive/`: original April derivation and Appendix A v4 recovered from
  laboratory storage. They contain superseded claims and are historical only.

For an independent Gaussian linear interpolant the score readout is
`(t*v-x)/((1-t)*sigma^2)`, with velocity v, t<1, norm cap 1000.
The identity is not generally the score of an arbitrary learned ODE, and the
aligned/retracted production path needs separate justification.

## Environments and storage

The work-correction development branch is specified in
`notes/nonequilibrium_framework.md`: `cfm_mol/nonequilibrium.py` implements
existing finite-path importance weights and an AIS teacher, while
`cfm_mol/clamped_work.py` connects frozen conditional FM velocities to
Gaussian proposals in an orthonormal COM-free basis. This changes the sampler;
it does not retrospectively correct archived samples. Explicit spin, target
confinement and ESS diagnostics are required. No molecular advantage is yet
established, and AIS/SNF/FEAT identities are not claimed as new.

The eSEN checkpoint is available at
`/n/holylabs/woo_lab/Lab/yulili/bgfm/checkpoints/omol25/esen_sm_conserving_all.pt`
and has passed a CPU H2 energy/force smoke in `envs/omol25`. See
`research/evidence/oracle_availability.json`. The previous missing-checkpoint
statement was caused by checking an obsolete storage path. The public OMol training archive is now available under
`/n/holylabs/woo_lab/Lab/yulili/bgfm/raw_data/omol25/v250514/`. Exact replay has
restored metadata for every legacy record, with an integrity-checked read-only
index and original energy/charge/spin sidecars. Official validation recovery and
overlap auditing are also complete, with 664 development and 722 reserved
conditions. Reserved method outcomes remain unqueried; explicit source links
do not exhaust every parent-trajectory relationship. Read STATUS for full scope.

Do not merge the environments: flowmol (torch 2.2 + DGL + Lightning) is for
training/inference; omol25 (torch 2.8 + fairchem) is for preprocessing/oracle
queries. Never install fairchem into flowmol.

The active checkout is `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`.
Its new runs are stored locally under that laboratory path. Its environments
and processed-data links point into
`/n/holylabs/woo_lab/Lab/yulili/bgfm/`. The older ryl_lab tree contains the
April code and notes; do not assume it is the latest training checkout.
The repository's storage restriction prohibits writing into home. Stage
changes and builds in /tmp unless the user explicitly allows small code
edits in the current checkout; persist large outputs in laboratory storage.
Use `PYTHONNOUSERSITE=1` so a user-installed NumPy 2 does not break torch 2.2
and wandb. Set `PYTHONDONTWRITEBYTECODE=1` and a /tmp `MPLCONFIGDIR` for audits.

## Reproduction commands

From the intended checkout, with `FLOWMOL_PY` pointing to envs/flowmol/bin/python
and `PYTHONPATH` including that checkout (the environment has editable installs):

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bgfm-mpl \
  "$FLOWMOL_PY" -m pytest tests/ -q -o cache_dir=/tmp/bgfm-pytest
python scripts/audit_iclr_evidence.py --out /tmp/bgfm-evidence
bash paper/build.sh /tmp/bgfm-paper-build
```

The evidence script needs only Python's standard library and checked-in files.
It verifies the original scalar while joining re-score records to energies,
uses checkpoint intersections across resolutions, drops reference geometries,
and applies the archived 93-parent mask. That mask is not an independent
molecular-identity split audit. It reports every seed and relaxation failure.
The build script exits nonzero for errors, unresolved citations, or >9 main
pages; passing this formatting gate does not certify scientific readiness.

## Data and runtime details

Read OMol25 labels with `atoms.get_potential_energy()` and `atoms.get_forces()`;
they live in `atoms.calc.results`. `n_atoms_histogram.pt` is `(values, counts)`.
Keep `python -u` and `PYTHONUNBUFFERED=1` in SLURM launchers. The valency JSON
is required by SampleAnalyzer, whose bond-free RDKit failures are guarded.
Use only within-parent energy variance; cross-batch offsets are confounded.
Charge clipping and missing spin in the historical preprocessing restrict
claims; a new chemistry-wide study needs preserved metadata and identity splits.
