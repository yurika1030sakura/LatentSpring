# Response to the supplied manuscript review

The supplied review is preserved in `review/review_20260915.md`. Its priorities
now govern this follow-up: demonstrate the complete method, test training-repeat
and budget dependence, and center the effective source and physical update.

## Completed matched experiment

The missing Gaussian physical-update cell is trained and independently audited.
Both source models use the same original physical recipe:256 native-source FIT
starts per continuation on the same eight training compositions, eight local
particles, no replenishment, and two1000-step physical/replay students. Each
source has its own generated anchors. The maximum preparation budget is shared;
actual eligible-anchor counts and query costs are reported separately.

GFN2 joint graph/force<=5 yields on1536 raw attempts per model are:

| Model | Graph validity | Joint quality yield |
|---|---:|---:|
| Gaussian |44.53%|35.68%|
| Gaussian + paired physics |46.61%|40.625%|
| Harmonic |52.73%|44.53%|
| Harmonic + paired physics |53.45%|47.59%|

Physical training improves joint yield within both sources in both continuations.
The harmonic-minus-Gaussian contrast with physics is+6.97pp pooled
(composition95 interval[3.26,10.61]), with individual contrasts+16.28/-2.34pp.
Raw graph-validity gains with physics are positive in both continuations.
Gaussian physical adaptation also lowers paired, jointly valid energy and force
under both eSEN and independent GFN2. Thus there are distinct source and update
benefits, while source variability remains important; positive interaction or
universal source superiority is not claimed.

The new cell adds512 FIT starts,1536 evaluation outputs,8348 eSEN queries and1632
GFN2 attempts. One GFN2 failure concerns a graph-invalid output. All valid outputs
and reference inversion checks succeed. Audit46647903 reconstructs teacher
proposals/support, uniform candidate selection, parameter arithmetic, structural
assays, parity averages and raw GFN2 parsing. The evidence is
`research/evidence/source_physical_factorial_audit_v1.json` and its NPZ; paired
contrasts are reproducible with `summarize_source_physical_factorial.py`.
Ledger v16 includes this completed study. The three old cells are reused; this
is a fixed follow-up on the already evaluated24-composition panel.

## Manuscript changes

The abstract and main text now center the structured source, replay subtraction,
the same-norm control and the complete four-cell physical-quality comparison.
The full force-threshold curves appear in the appendix. The source determinant
is explicitly an analytic property, not evaluated in primary FM training; the
physical teacher has its separate Gaussian anchor reference q_A.

HarmonicFlow/FlowSite is cited from its ICML2024 PMLR record:
https://proceedings.mlr.press/v235/stark24a.html. The distinction is lack of a
supplied chemical graph, not first use of harmonic/self-conditioned flow matching.
The independent EGNN comparison now explicitly states that those runs omit
self-conditioning and physical adaptation. It is not presented as a comparison
of the full pipelines.

The rotor58.5% number is removed from the abstract. Complete work, rotor recovery
and curvature results remain as theoretical extensions and boundary analyses in
the appendices. No further rotor/curvature study is planned. The current local
build has9 scientific main pages,26 total pages and25 citations.

## Active experiments and remaining gap

Three additional paired source continuations retain1000/3000/6000-step checkpoints
and every outcome. Together with the original two there will be five stochastic
continuations of one pretrained model on two fixed3000-row blocks. These are not
five independent pretraining runs or a controlled dataset-size study.

The unstarted normal-GPU array46643602 was cancelled. Dispatcher46654304 waits
for GAGA seed1, then submits the unchanged original136257b experiment source to
gpu_test, running seeds2,3,4 sequentially. Its GPU job will be recorded in
`runs/source_replication_v1/gpu_test_submission.json`. Audit46654303 is held until
the dispatcher assigns that GPU dependency and releases it. The earlier audit
46647904 was automatically cancelled with its old parent; no computation was lost.
Results/checkpoints are in netscratch via `runs/source_replication_v1/results`.

GAGA seed1 remains46635650_1. Job46645569 finished the factorial and resumed GAGA
seed0 from its saved distance checkpoint. Its small interrupted tree prefix is
preserved in the logs as discarded execution, not unique training progress.
Confirmation dispatcher46636422 depends on the live seed jobs. No GAGA method
choice or numerical protocol was changed by the scheduling switch.

A comparison where the strong GAGA baseline also receives matched physical
adaptation remains open. Neither the primary-backbone factorial nor the ongoing
raw-GAGA feedback test alone closes that gap. New repetition and GAGA outcomes
must be audited before the paper's claims are expanded.
