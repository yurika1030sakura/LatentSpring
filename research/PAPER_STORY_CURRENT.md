# Current paper story — September 12, 2026

The problem is to propose useful molecular connectivity changes with compatible
3D geometry and computable reverse probabilities. A fixed OMol25 flow-matching
initializer supplies coordinates with unknown finite-step density. A separate
corrector couples connectivity, radii and angular placements, using exact MH
correction for the stated restrained inversion-averaged eSEN target.

The latest candidate learns a conditional placement-energy function from fixed-
context potential differences and tangential forces. Its masked context predicts
radial interaction curves, while candidate directions are evaluated on feasible
circle arcs with explicitly normalized interpolants. Complete joint proposals
include both circle preimages, both root orders and Cartesian radial factors.
This scalar candidate is implemented and trained; it has not yet demonstrated
an advantage in molecular chains.

Four800-step models improve nonlocal work prediction on internally withheld
parents and compositions by about19--24% against the physical score. Independent
rebuild and metric replay confirm this signal. The physical site-arc construction
already improves one training-panel chain readout at equal oracle calls, with
extra runtime. Neither result establishes equilibrium, blind generalization,
competitive learned-generator performance or standalone AI novelty.

The intended AI contribution is useful conditional learning for joint molecular
connectivity and3D placement, under explicit geometric support and realistic
costs. Classical regrowth, self-learning MC, force fitting and learned decoders
have substantial prior art. The next complete-chain comparison and appropriate
baselines must show what this particular construction adds.

The development manuscript `paper/angular_working.tex` includes the older learner's
adverse strong-control results, the physical arc appendix, and the new scalar
learning table with its internal-validation and cost limits. The current reviewed
PDF is `runs/verification/conditional_nonlocal_20260912/main.pdf`. See
`research/CONDITIONAL_NONLOCAL_STATE_20260912.json` for exact evidence and costs,
and `notes/nonlocal_arc_prior_art_20260912.md` for the current novelty boundary.
Prior story: `notes/archive/paper_story_current_through_geodesic_physics_20260912.md`.
