# Joint graph/geometry decision after completed full-cost controls

September12, 2026. The ICLR goal remains active and unachieved.
Canonical evidence: `research/evidence/joint_geometry_decision_v1.json`.

## What is implemented and verified

`cfm_mol/joint_chemical_geometry.py` selects unlike terminal leaves at distinct
anchors, draws two log-normal radii and normalized autoregressive directions,
checks the desired perceived graph and evaluates the complete reverse density.
The log-radius Cartesian factor is r^-3 for each root; the old deterministic-map
Jacobian does not apply to newly sampled coordinates. A fair augmented root order
preserves permutation symmetry. Failed support checks are single-draw self loops.

Eighteen targeted tests passed, including the conditional COM-volume chart,
radial known-target test with a missing-Jacobian negative control, sequential
context reconstruction, O(3)/permutation covariance and existing angular kernels.
This is not a claim that the entire repository test suite was run.

Pilot46110667 completed ten arms. Full-cost continuation46111148 completed six
physical-control arms with unchanged states and RNGs. CPU audit46111921 replayed
every continuation and checked the complete immutable prefixes and independent
MH densities. Maximum coordinate replay error is0. All requested/acknowledged
physical queries agree. A local audit process received SIGTERM before producing
a final artifact; its unknown cause and successful CPU-queue recovery are recorded.

## The positive result and the control that changes its interpretation

The joint tensor pilot reaches the diagnostic reference connectivity for all four
generated development parents in both replicas. The difficult third parent's
first hits are steps91/23. The earlier fixed-geometry tensor sampler failed both
replicas. This supports retaining joint graph/geometry updates.

However, the physical site prior also reaches all four when its chains are
continued. Preparation-inclusive first-passage costs are:

| Method | Replica0: step / total raw calls | Replica1: step / total raw calls |
|---|---:|---:|
| Learned tensor |107 /11296 |67 /11076 |
| Physical site prior |751 /6380 |287 /2994 |

The learned method uses fewer online microsteps, but inherits9660 additional
training/source calls. The site prior wins the measured cumulative-cost comparison.
Deterministic controls still fail the difficult parent; uniform joint geometry
reaches none of the diagnostic graphs. Final control budgets are within0--6 calls
of the corresponding tensor budgets. No step cap binds.

This round spends17014 new pilot calls plus58946 continuation calls =75960 total.
The previously paid training table is reused; no new physical training queries
were made. The current four-parent experiment does not measure amortized reuse.
The reserved evaluation outcomes remain unused.

## Distribution and transfer failures remain

Final-half restrained-potential rank/folded split Rhat is1.802/1.906 for tensor
and1.126/1.208 for the longer site controls. Shape diagnostics are also poor.
Successful first passage and low final energies therefore do not qualify the
300K target distribution. Keep this limitation in the abstract and conclusions.

The next eligible composition has17 atoms and O/P/Cl, absent from the current
guide's training composition. A geometry-only screen on its first32 supported
training parents gives tensor44/128 and0/128 valid endpoints versus site101/128
and96/128. No force/energy calls or fitting occur in that screen. The strict source
loader reads both streams for provenance; only training positions enter proposals.
This is a transfer warning, not an independent energy-distribution experiment.

Using a normalized two-vMF mixture changes the tensor's actual angular score:
training MSE575.2/602.7 rather than the fitted quadratic score's439.7/475.0. The
vector ablation is unchanged. This is an identified objective/decoder mismatch,
not a proven causal explanation of the molecular result.

## Next decision

Do not expand the frozen guide or present a total-cost advantage. Preserve the
exact coordinate framework and the strong physical prior. Implement and test the
bounded actual-density/residual-prior experiment in
`notes/normalized_joint_learning_next.md`; then require independent compositions,
prospectively measured reuse costs and distribution diagnostics.

The working paper is `paper/angular_working.tex`, currently six main-text pages.
It contains the joint derivation, full positive/negative results and a cost plot.
`paper/main.tex` remains the older study. Generic graph-editing MCMC, masked MH,
normalization, force fitting and directional mixtures are not standalone novelty;
MARS and MOSAiCS were added to the related-work record. Scientific submission
readiness remains false.
