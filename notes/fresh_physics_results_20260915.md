# LatentSpring: fresh validation and direct-generation priority

The frozen 24-composition confirmation is complete. Both existing trained
continuations improve raw graph support with the harmonic source. The preselected
force-update candidate improves raw energy and force under both eSEN and the
independent GFN2-xTB evaluator, passing the frozen cross-potential gate.
Scientific submission readiness remains false.

## What inference actually does

The network generates coordinates from composition and a harmonic latent-tree
source at 128 primitive denoiser calls per attempt. No per-output energy ranking,
force correction, or structural optimization is applied. xTB uses `--grad` and
never `--opt`; its input coordinates and atom symbols were checked against all
saved generator outputs. Physics enters the separate FIT teacher and learned
parameter update. Evaluator gradients do not modify the reported structures.

## Frozen new-composition results

Twenty-four new reference-qualified compositions (17–28 atoms), eight per size
bin, are selected by fixed metadata ranks from 256 reference candidates. All old
official candidate compositions and matches in the two verified processed corpora
are excluded. References are outside fitting and never initialize generation.
Each of four models makes 32 draws per composition in each existing continuation:
6,144 new raw outputs. These are new compositions, not new independently trained
models. All outcome failures remain in their appropriate denominators.

| Model | Continuation 1 /768 | Continuation 2 /768 | Raw graph support |
|---|---:|---:|---:|
| Gaussian | 332 | 352 | 44.53% |
| LatentSpring harmonic | 435 | 375 | 52.73% |
| Force update | 439 | 382 | 53.45% |
| Complete-work update | 426 | 383 | 52.67% |

Harmonic minus Gaussian is +8.20 percentage points, conditional paired 95%
interval [5.08, 11.33]. Force update minus harmonic is +0.72 points [-0.98, 2.34]:
this does not establish an additional validity increase from the physics update.

| Primary force update minus harmonic | eSEN | GFN2-xTB |
|---|---:|---:|
| Energy, eV/atom | -0.01308 [-0.01805, -0.00817] | -0.01168 [-0.01616, -0.00724] |
| Force RMS, eV/Angstrom | -0.34793 [-0.41498, -0.28478] | -0.35620 [-0.42983, -0.28714] |

These are conditional paired 95% intervals on common graph support, with SCC
success additionally required for GFN2. Both continuation signs improve for both
metrics under both potentials. All 24 cells per continuation have comparable
primary pairs. GFN2 has four failed generated calculations out of 6,144, all
outside graph support; all 96 reference calculations succeed. Costs are 12,384
raw eSEN queries and 6,240 GFN2 attempts. Both potentials are approximate. The
source-only physical advantage is heterogeneous across the two continuations.

Complete-work minus force-update energy intervals span zero under both evaluators.
Jarzynski remains an implemented, tested local-work comparison; its extra benefit
after useful transfer and a global Boltzmann output law remain unestablished.

## Raw failures and next essential work

Disjoint categories for the force-update candidate, out of 1,536 attempts:

| Category | Count | Fraction |
|---|---:|---:|
| Graph supported | 821 | 53.45% |
| Disconnected only | 474 | 30.86% |
| Overlap only | 3 | 0.20% |
| Both disconnected and overlapping | 7 | 0.46% |
| Geometry passes, graph perception rejects | 231 | 15.04% |

Including both-defect cases, disconnection affects 481/1,536 (31.32%). It is the
largest diagnosed failure category; graph rejection requires further diagnosis
and is not automatically evidence of any particular chemical defect. These are
descriptive observations after evaluation, with no change to assay or denominator.

The user's priority is directly useful output with little optional relaxation.
53.45% graph support does not yet meet that aspiration. Lower force alone does
not measure steps to convergence or prove proximity to a stable minimum. Any
claim of little optimization should report raw validity first, then a separate
matched capped-relaxation assay with coordinate change, graph retention, failures,
steps and cost. Original coordinates must remain the primary output record.

Next, develop one targeted connectivity improvement on FIT-only data, consulting
the already failed tree-context/product-coordinate/edge-feedback controls before
choosing it. Fresh24 outcomes are now observed: never use them for fitting or call
subsequent retuning untouched confirmation. Preserve original harmonic and physics
checkpoints. A fair independent-generator comparison remains essential.

The proposed new global stochastic-path/Jarzynski branch is deferred following
the user's clarification; no new implementation or job was launched. No broad
architecture or coefficient sweep is queued. Reserved722 outcomes remain unqueried.

## Reproducibility

Generation and eSEN: jobs46531691_0/1, completed. GFN2: jobs46531808_0/1, completed.
Producer snapshot: `c1596740e1021bfd9f80c93e281767b772a11cda`.
Audits: `research/evidence/fresh_physics_esen_audit_v1.json`,
`fresh_physics_xtb_audit_v1.json`, `fresh_raw_failure_breakdown_v1.json`.
Reproduce the last with `scripts/research/audit_fresh_raw_failures.py --project .
--out research/evidence/fresh_raw_failure_breakdown_v1.json` in the flowmol env.
No new model generation or physical queries occur during these audits.
Registryv12 has 84,160 evaluation NN outputs, 3,584 FIT outputs, 87,744 total NN
records, and 44,256 source/teacher eSEN rows; the 6,240 fresh GFN2 attempts are
counted separately. Older physical campaigns and derived projections remain
separately recorded in the registry chain.
