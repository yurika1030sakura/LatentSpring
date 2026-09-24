# Direct geometric correction inside the sampler

The first geometry-context fit produced almost no geometric gain. This separate
candidate gives the learned module direct control over coordinate velocity.
It is fitted to denoising targets from training references and is followed by
the original frozen physical correction at each sampler evaluation. Neither
the parent nor the physical-head weights are retrained.

A fixed hash selects4,096 rows from the original20,000 training references.
The declared coordinate-derived geometry subset screens these references;
every selection result is retained. Inferred graphs are used for reference
screening, never supplied as network inputs or supervised bond labels. The
evaluation conditions remain disjoint from training and the original baseline
outputs are reused. No new quantum teacher queries are needed.

At progress p in[0.4,0.95], the reference receives centered Gaussian coordinate
noise with standard deviation0.06+0.30(1-p) Angstrom;15% of examples are unchanged
references with zero recovery target. The velocity target is the reference-minus-
corrupted displacement divided by1-p, with a common scale enforcing the declared
4p^2 per-atom bound and preserving centering. Hydrogen errors receive weight0.25,
heavy-atom errors weight1. This is bounded recovery regression, not a claim of
an exact equilibrium score or Boltzmann sampler.

The7,747-parameter field uses symmetric pair features and three antisymmetric
vector directions. The moment version includes local directional first/second
moments, supporting corrections beyond the pair axis. Its equal-parameter radial
control replaces angular invariants and vector directions by radial features.
Both are initialized to zero and receive the same batches and perturbations.
Message normalization gives zero total correction and a per-atom velocity norm
at most4p^2; no final-geometry guarantee follows from that bound.

Sampling first changes the provisional endpoint through the learned geometric
velocity, then applies the existing physical head at that updated endpoint.
The field enters every integration step; there is no output optimization.
Each draw uses128 backbone,64 geometric-field and64 physical-head evaluations.
All raw coordinates undergo the same graph, geometry and fixed-coordinate GFN2
checks as the controls. The main64-composition benchmark is not used for selection.

Each of two fit/field seeds has a radial and a moment arm, each trained for10,000
updates of batch32. The budget is40,000 small-field optimizer updates,1,280,000
field training-example forwards,1,536 new outputs/GFN2 attempts, and zero parent
training forwards or new eSEN calls. The768 frozen controls are reused. The new
allocation requests at most2 GPU hours, within the original overall18-GPU-hour
ceiling alongside the other current studies.

Ten targeted tests pass across the shared context and direct-field code. They
cover symmetries, centering and bounds, target replay, unchanged physical output
at zero initialization, and actual trainability of the correction directions.
The running earlier studies retain their immutable code and protocols. This
candidate has no generation-performance claim until its paired outcomes complete.
