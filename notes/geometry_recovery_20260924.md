# Geometry recovery from training coordinates

The round2 audit identified heavy-atom fragmentation and local geometry as the
main remaining defects. The first experiment keeps the harmonic source, EGNN
architecture, distance self-conditioning, midpoint sampler, and existing physical
head. Only backbone continuation training changes. This separates a train-time
intervention from test-time optimization.

The two original fitted FM parents each receive three paired continuations:

1. Replay: original CFM updates alternate with late endpoint regression on the
   original training interpolation.
2. Recovery: the same late states receive masked atom perturbations and a
   displacement of a spatially defined region. The target is the original
   training reference after its existing rotation/permutation coupling.
3. Recovery with local geometry: the same corrupted states additionally receive
   smooth reference-relative pair-distance and neighbor-angle supervision.

All arms use the same 20,000 OMol25 references, batches, initialization within
each fit, two backbone evaluations per update, learning rate, and update count.
There are 4,000 updates of batch32 per arm. The final EMA is used. The additional
geometry operations have their own cost; matching backbone calls is not a claim
of identical FLOPs or wall time. No chemical bond labels are added. Reference
distance neighborhoods are continuous geometric weights. The auxiliary task is
endpoint recovery, not an assertion that perturbed states have the original CFM
path's velocity target. Original CFM supervision remains on alternate updates.

The physical head stays fixed at its original strength4. It is applied to every
evaluation model, including the frozen and continued controls. Every output uses
128 backbone and64 head evaluations, no hydrogen readout, and no energy queries
or geometry optimization during sampling. GFN2 is evaluated afterward at those
unchanged coordinates. Every attempt remains in all-output denominators.

The development panel comprises 24 validation compositions selected by a fixed
hash order, 12 with17--28 atoms and12 with29--40 atoms, excluding the four round2
illustration compositions. It is disjoint from training, but is explicitly a
development panel: we do not claim it was never used in earlier project work.
Each fit/method draws16 samples per composition. The primary64-composition
benchmark is not consumed for this pilot. Future confirmation must use a
separately specified panel and report its prior-use status.

Selection uses all-output closed-shell geometry-subset yield, its conjunction
with force RMS<=5 eV/Angstrom, and graph-valid distinct-connectivity yield.
Both recovery candidates must beat both the frozen and continued controls in
both fits for the geometry and geometry-plus-force point contrasts. A97.5%
composition-bootstrap interval addresses the two candidate comparisons; crossed
fit/composition intervals and all individual fits are also reported. A gain in
force scores alone does not qualify. Geometry failure categories remain visible.

This is a bounded first stage. It tests whether better recovery supervision
helps before increasing architectural complexity. Successful recovery motivates
an explicit multi-atom geometry network and separate confirmation. A failed
pilot must retain every outcome and be used to distinguish insufficient training
signal from insufficient model expressiveness; it does not justify silently
selecting a different checkpoint or retuning on the primary benchmark.

The main manuscript and its existing results remain unchanged until completed
evidence supports a revision. Geometry corruption and equivariant recovery have
prior art, including FlowMol3's late-stage geometry distortion:
https://arxiv.org/html/2508.12629v1#S2.SS6.SSS3

Protocol: `research/evidence/geometry_recovery_v1.json`.
Code: `cfm_mol/geometry_recovery.py`, `scripts/research/run_geometry_recovery.py`.
Audit: `scripts/research/audit_geometry_recovery.py`.
Four targeted tests cover geometric invariance, gradients, corruption replay,
and real two-pass EGNN backward passes in all three training branches.
