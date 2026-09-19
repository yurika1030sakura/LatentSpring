# Broad physical endpoint learning

The earlier four-graph/three-composition prototype improved its training tasks
but did not establish physical-quality gains on unseen compositions. This study
tests direct physical-target learning over128 OMol25 TRAIN compositions. It
changes neither the harmonic source nor the FlowMol backbone.

## Frozen comparison

The protocol is `research/evidence/broad_physical_endpoints_v1.json`.
We select32 distinct compositions in each8--16,17--24,25--32,33--40 atom bin by
salted metadata hash. Each uses one original reference from the existing3000-row
TRAIN block. No energy or generated outcome is used to choose references.

For reference A, R(A) is the last accepted point of a bounded, graph-preserving
Cartesian L-BFGS search under the inversion-averaged eSEN potential. All atomic
coordinates can move. The search has a0.05-Angstrom per-atom step cap,96 accepted
step opportunities,160 energy/force evaluations, and max-force tolerance0.1 eV/A.
Unsupported trial graphs are rejected before physical evaluation. Energy queries,
failed line-search proposals, accepted steps and termination reasons are saved.
Budget exhaustion or a stalled search returns the last accepted point, possibly
the reference; convergence is reported separately. No composition is discarded.
This is a training-data transformation, not a new optimization algorithm.

The reference target is the uniform empirical mixture over A_i, with0.005-Angstrom
centered Gaussian smoothing. The physical target is the same mixture over R(A_i),
with identical smoothing. This preserves the prescribed composition frequencies.
It is an empirical target, not a300 K Boltzmann ensemble; no new work-weighting
benefit is claimed. The existing Jarzynski-weighted branch remains separate.

Two fine-tuning seeds (52001/52002) start from the same audited5.90M harmonic
checkpoint. Each fits both original and corrected reference targets for2000
updates at batch1, using the same row choices, source noise, symmetry augmentation,
optimizer and two-pass FM objective. There is no parameter subtraction.

Six evaluations include the unadapted parent and both fine-tuned models for each
seed. The fresh panel has16 compositions, eight each in17--28 and29--40 atoms.
They were previously qualified but never generated; the panel excludes both
processed corpora and the previous64/32/16 panels. Earlier development, reserved,
and24-composition families were already excluded by the pool construction.
Every model returns16 raw outputs per composition with128 backbone calls and no
terminal noise, energy ranking or geometry optimization. No checkpoint selection
or hyperparameter tuning uses these outcomes.

The main contrast is corrected-reference minus original-reference fine-tuning in
all-attempt graph-valid GFN2 force-RMS<=5 yield. Both are also compared with the
parent, with all frozen thresholds reported. Composition intervals condition on
the two fitted runs; these are not two independent pretraining runs. The study
does not compare directly with GAGA and does not isolate coverage from the changed
training budget of the earlier500-step pilot.

## Completed target checks

All128 selected targets were retained. eSEN force-RMS medians fell from2.00528 to
0.05038 eV/A;124 references reduced force RMS and110 reached max force<=0.1.
Four cases increased RMS force along energy-decreasing graph-preserving searches;
they remain in the bank. Every query and search decision was numerically replayed.
The teacher used6576 raw eSEN evaluations, including inversion pairs.

An independent GFN2 check covers all128 references and all128 corrected targets.
Median force RMS is1.75577 versus0.39741 eV/A. The fraction meeting force RMS<=1
is22.656% versus89.844%, and the<=5 fractions are96.094% versus100%.
These256 records are training labels, not neural outputs. Their scores do not
establish improved generation; the matched model comparison answers that question.

Jobs and current completion are recorded in
`research/BROAD_ENDPOINTS_STATE_20260919.json`. All raw artifacts are under
`runs/broad_physical_endpoints_v1`. The published manuscript remains unchanged
until the generator results have been checked.
