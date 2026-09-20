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
