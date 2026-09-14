# Reference-qualified monomer results and AI contribution decision

The prospective monomer evaluation is complete. All10,752 generated source draws
and structural outcomes replay, all21 reference assays replay, and actual training
rows match within each continuation. The task/readout alignment is now concrete;
it does not establish an extra benefit of the learned source networks.

## Prospective task and separation

Neutral singlet organic structures with8--40 atoms were selected using original
references and metadata before generated outcomes. All54 compositions in four
declared earlier panels were excluded. All78 metadata-eligible references were
retained for qualification;31 passed, and a fixed hash selected21 with5/8/8 in
three size bins. Reference coordinates were absent from the generator manifest
and never used for fitting or initialization. The old broad panels are preserved.

An exact count-vector check finds zero equal compositions in3,902,107 training
and39,415 legacy-validation records from the two checksum-verified processed
files. The additive-hash screen is followed by exact checks, with deliberate
collision and real-corpus positive controls. This is separation from those
corpora, not a certificate for arbitrary unrecorded checkpoint pretraining or
complete trajectory identities.

## All models, no source selection by seed

| Continuation | Gaussian | Shell | Harmonic | Learned node | Learned pair |
|---|---:|---:|---:|---:|---:|
| 1 |581|627|575|603|561|
| 2 |521|557|598|—|—|

Counts are graph-check passes out of1,344 per model. Node/pair source models exist
only for the first continuation; no second learned-prior replicate is implied.

Shell minus Gaussian is+3.42 percentage points, paired95% interval[-0.15,6.99],
and+2.68 pp[-0.74,6.10]. Descriptive size-stratified composition intervals are
[0.52,6.55] and[-0.52,5.73]. The two summaries should both be retained; do not
choose the favorable interval to claim a stable graph advantage. Harmonic minus
Gaussian is-0.45 pp in the first continuation and+5.73 pp in the second, with a
positive interval only in the latter. No source is a universal winner here.

Node minus shell is-1.79 pp[-5.21,1.64]. Pair minus shell is-4.91 pp
[-8.41,-1.41]; its descriptive composition interval is also negative. Thus the
source-likelihood learner does not establish an extra generation benefit on the
reference-qualified task. This cannot be explained solely by poor reference-assay
coverage, which was resolved prospectively. It still does not identify a unique
cause: objective mismatch, capacity, optimization and data coverage remain possible.

The chemical readout is operational graph support. It does not certify physical
stability or thermal populations. All intervals are marginal, without multiplicity
adjustment; composition bootstrap is descriptive. No energy result or final
likelihood/ESS/Boltzmann claim is added by this experiment.

## What to claim, and what to do next

There is an implemented AI-method candidate: a learnable, explicitly normalized
spatial source without supplied chemical bonds, integrated with molecular FM.
The existence of this object is stronger method content than an auxiliary-loss-only
proposal. However, general tree ensembles, harmonic priors and latent conditioning
are prior art, and sufficient distinctiveness/usefulness for ICLR remains unproved.
Do not claim that the extra neural source or static context module has won.

Do not simply scale the same coordinate-NLL heads or static context adapter on
these observed outcomes. A focused next hypothesis is source learning from actual
generated-sample utility with the decoder held fixed, so the learning objective
addresses sample effectiveness directly. `notes/source_utility_learning_brief_v1.md`
states the exact source-space importance identity and a possible analytic
distribution-shift control. It is a proposal only: no utility-trained model,
training bank or positive result exists yet, and its generic identities are not
claimed as new theorems. A dedicated prior-art check and bounded actual/shuffled-
utility comparison would be required.

## Reproducibility and costs

Jobs46415396_0/1 complete0:0 in34:26/20:28. Eight frozen checkpoints produce10,752
new outputs, with no new training and zero new physical-oracle queries. The
source-restoration/tree tests pass17 cases. Source draws, saved prior laws,
reference qualification, matching training rows and final structural readouts
replay. Full optimizer trajectories and final neural integration are not rerun.

Core artifacts: `monomer_panel_v1.json`, `monomer_qualification_v1.json`,
`monomer_overlap_v1.json`, `monomer_overlap_controls_v1.json`, and
`monomer_evaluation_audit_v1.json` under `research/evidence/`. Figures are in
`research/figures/monomer_v1/`. The internal manuscript includes these results.

Total tree/monomer outputs are30,208. Keep them, all prior evaluated cohorts,
the78 candidate reference geometries, and722 reserved outcomes outside fitting.
The ICLR goal remains active and scientifically unachieved.
