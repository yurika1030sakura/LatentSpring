# Current paper story — September 12, 2026

The current paper is `paper/angular_working.tex`, **Learning Geometry for
Reversible Molecular Graph Exchanges**. It is a development draft with verified
methods and adverse controls; it is not an established competitive ICLR method.

A fixed OMol25-trained flow-matching initializer supplies atom- and electronic-
conditioned coordinates. Its finite-step source density is unknown. Subsequent
reversible graph/geometry proposals have explicit forward/reverse conditional
probabilities and Cartesian volume factors, permitting MH correction for the
stated restrained, inversion-averaged, algorithmic-support target. The learned
component predicts geometric proposal densities from masked molecular context.
This is an implemented method candidate; generic MH and force fitting are not
standalone novelty claims.

The initial 72-parent energy benefit against the concentration-10 physical prior
is independently corroborated by GFN2 on those original endpoints. It fails to
remain established with stronger physical widths. Frozen transfer to six further
compositions also lacks a demonstrated average benefit. These limits are stated
in the abstract and main results; stronger physical endpoints have not received
that original GFN2 check. Lower finite-budget energy is not a calibrated energy
distribution or an equilibrium ensemble.

The next research question is whether identifiable conditional training signals
and calibrated directional predictions can yield useful transfer across molecular
contexts. New training data now cover six additional supported compositions, with
two zero-support compositions preserved. Data preparation and a local fitting
identity do not establish this proposed AI contribution. The bounded next design
is `notes/conditional_learning_after_sharp_controls_v1.md`.

Read `research/SHARP_CONTROLS_AND_TRAINING_STATE_20260912.json` for exact evidence,
`research/NOVELTY_DECISION.md` for claim boundaries, and
`research/CLAIM_AND_BENCHMARK_SCOPE.md` for the intended submission standard.
The previous entropy-refinement paper story is archived at `notes/archive/paper_story_current_through_transfer_checkpoint_20260912.md`.
