# Shared-backbone physical correction

The complete FlowMol system now improves with a small learned physical head.
The next scientific question is whether that training rule transfers to the
independently trained EGNN, and how it compares with giving GAGA the same
physical supervision. Protocol `matched_connection_v1.json` fixes this study
before sampling. Its results do not replace the earlier complete-system study.

## What the head predicts

At each sampling state, the parent predicts a provisional final geometry. Call
the current coordinates `x`, that provisional geometry `H`, and the fraction of
the sampling schedule already completed `p`. For flow matching, `p` is the flow
time. For truncated GAGA, `p = 1 - t/t_max`, where `t` is the remaining diffusion
noise time and `t_max = 0.65`. These are normalized schedule positions, not a
claim that the two algorithms follow the same physical process.

The pair head takes `x`, `H`, atom identities and `p`. It returns a centered
vector `A` with units of Angstrom per unit normalized progress. The implied
change to the provisional final geometry is `(1-p) A`. Its per-atom magnitude
cannot exceed `2 p^2 (1-p)` Angstrom. This bound is on the instantaneous
provisional endpoint change, not on the final sampled geometry or its energy.

For FM, the parent velocity `v` defines `H = x + (1-p) v`. Adding `A` to `v`
therefore gives the desired endpoint change directly.

For GAGA, the predicted noise `epsilon` defines
`H = (x - sigma(t) epsilon) / alpha(t)`. To produce the same endpoint change,
the noise prediction receives `-alpha(t) (1-p) A / sigma(t)`. Adding an FM
velocity directly to GAGA's noise prediction would use the wrong sign and
scale. The native GAGA stochastic transitions and final observation noise are
retained. At zero noise time, the added noise correction vanishes because
`1-p = 0`; the fixed schedule has a strictly positive noise precision.

## Training and comparison

The two parent families have the same 2,381,566-parameter EGNN and previously
matched training-example forward counts on the same 20,000 OMol25 rows.
Each independently fitted parent receives the same 128 TRAIN compositions,
two native trajectories per composition and two late provisional states per
trajectory. Every state is retained. Forces come from fixed-coordinate eSEN
queries at both `H` and `-H`, with inversion symmetrization and center-of-mass
removal. There is no graph-validity filter on the teacher bank.

A bounded small displacement along that force gives the target endpoint
change. Dividing by `1-p` gives the common target for `A`. The same whole-state
scale enforces the head's per-atom bound if needed. Each head trains for 20,000
updates on 96 compositions; the other 32 TRAIN compositions measure target
prediction transfer. With the common 16-element vocabulary, the head has
7,106 parameters. The 8,178-parameter FlowMol head uses the larger 83-element
embedding vocabulary; its hidden architecture is otherwise the same.

Eight separate validation compositions choose a single strength for each
algorithm across both fits, from `0, 0.25, 0.5, 1`. Selection maximizes the
fraction of all outputs that have an inferred valid graph and GFN2 RMS force
at most 5 eV/Angstrom. Ties choose the smaller strength. A new 16-composition
test panel is generated only after the selection is audited and frozen.

The primary contrast is corrected FM minus its own parent. Corrected FM versus
equally supervised GAGA and the difference in their improvements are reported
separately. All intervals resample compositions while retaining both model
fits. This prevents the full-model architecture gap from being attributed to
the physical head.

Both methods use 128 backbone calls, but FM uses 64 head calls and GAGA uses
128. Training supervision and head-update budgets are matched; total FLOPs and
wall time are not claimed equal. Sampling makes no energy calls, optimization
steps or energy-based selections. This study tests a learned local force
correction; it adds no claim of Jarzynski-specific neural improvement or global
Boltzmann sampling.

## Required checks

The production FM and GAGA samplers return bitwise identical samples and source
coordinates with a zero head or passive observer. Tests also verify the
endpoint conversion, rotation/permutation symmetry, zero center-of-mass
correction, head-only gradients and exact backbone/head call counts. All parent
tensors, including GAGA's noise schedule, remain frozen. Independent auditing
replays target construction and head-validation errors, checks raw output
hashes, re-runs graph assessment, and parses fixed-coordinate xTB gradients.
