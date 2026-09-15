# Stronger generator evidence and broader compositions

The user authorized continuing both priorities after an explicit comparison with
recent ICLR papers. The current LatentSpring manuscript remains the last verified
release until new evidence is audited. Existing fitted models remain frozen.

Two prospective studies are being prepared:

1. A 64-composition panel, 16 in each of the 17–28, 29–40, 41–52 and 53–64 atom
   strata. The initial pool uses 192 metadata-selected candidates per stratum.
   Reference qualification uses the existing graph/connectivity/overlap assay;
   exact composition matches in both processed corpora are removed before ranking.
   Selection never uses model outcomes or energies. The original Gaussian and
   harmonic models will be evaluated at the same 128 primitive calls.
2. An independent EGNN benchmark using 20,000 OMol25 training rows and 512 internal
   validation rows separated by composition hash. All four arms start from the same
   random weights within a seed, use the same size-balanced minibatch stream,
   capacity, optimizer and fixed training budget. The arms are Gaussian FM,
   harmonic FM, conditional EDM and position-conditional GAGA truncation. Two
   independent initialization seeds are planned. The FM comparison isolates the
   source effect on a different backbone without pretraining confounds.

The GAGA adapter follows its published Gaussian variance propagation and uses the
published GEOM truncation at 650/1000; this transferred hyperparameter is fixed,
not selected on evaluation molecules or claimed to be OMol25-optimal. Coordinate
variance is estimated from training data only. Its scientific reference is
https://proceedings.iclr.cc/paper_files/paper/2026/file/08de3c1eb4adc7ac3c1949048422d895-Paper-Conference.pdf
and inspected upstream revision is 89c72c9b4bf774805bc6edc715d4cca1b62721a7
(https://github.com/QuJX/GAGA). The adapter is implemented from the equations,
using the already pinned official EDM backbone; no GAGA source is vendored.

No new bond labels, energy labels, geometry optimization, or full OMol25 training
are introduced. Models retain max_atoms=200; the experimental training subset
covers 8–64 atoms. Validation readouts diagnose learning and do not select a
checkpoint. Generation uses the EMA model at a fixed final update. Scratch
training is an additional controlled experiment and does not replace the
pretrained LatentSpring model in the manuscript.

Pool preparation job 46558599 completed. The first panel/data attempts
46559185/46559186 stopped before extraction because torch 2.2 requires a string
filename for mmap loading. The Path arguments were corrected; original logs and
output directories are retained. No neural outputs or physical calls occurred.

Four adapter tests pass: the harmonic sampler matches enumerated three-atom
mixture covariance; initialization is reproducible and every objective has finite
nonzero gradients; diffusion matches the existing conditional sampler; and the
EGNN vector respects rotations. These are implementation checks, not evidence of
competitive generation.

The expanded pool now yields 125/119/72/30 qualified references in the four size
strata. All qualified candidates have zero exact composition overlap in both
processed corpora. The final 64-composition panel is frozen. Training preparation
selected 20,000 rows (8,323 compositions) and 512 internal validation rows
(299 compositions), with no composition overlap between those partitions or the
new panel. No scaffold-tree filter was applied to these training rows.

The formal independent benchmark uses batch32 and30,000 updates (960,000
training presentations per arm), two independent initializations and four methods.
All arms share the batch stream within each initialization. The train-only mean
intrinsic coordinate variance for GAGA is3.0232257865777212 A^2. Fixed EMA outputs
at the final update are evaluated; validation at5,000/15,000/30,000 diagnoses
learning without model selection. The128-call comparison is primary; full EDM
(1001) and truncated GAGA (651) schedules are also retained.

GPU probes exercised all objectives on actual40- and64-atom FIT structures. The
larger batch32 probe uses at most6.72GB on an A100 MIG device. These probes are
engineering runs; no benchmark-generation result has been inspected. Checkpoints
include optimizer and EMA states for scheduler recovery. Completed generated
files are reused with model/condition/seed/hash checks during recovery.

The frozen-model comparison started as array46562196 on gpu_test. Larger molecules
require substantially more inference time than the earlier17–28-atom panel.
Requests to extend the running jobs' one-hour limit were rejected by Slurm.
A per-composition continuation retains original raw samples and their logged
timings, and generates only missing compositions after the original jobs stop.
Missing timing, if any, is explicitly marked rather than reconstructed as an
exact measurement. Original logs and interrupted fragments remain archived.

The independent-training array46562197 remained entirely pending. It was held
and cancelled before training, then replaced by46562802 with compatible
A100/H100/H200 node features. The same immutable computational source, protocols,
output directories and budgets are used. Hardware availability changes the
allocation; it does not select methods, data or outcomes.
