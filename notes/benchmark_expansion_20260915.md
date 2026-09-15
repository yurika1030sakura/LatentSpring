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
