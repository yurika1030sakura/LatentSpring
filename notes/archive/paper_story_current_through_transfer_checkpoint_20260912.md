# Current paper direction — September11,2026

**Active reconstruction:** read `RECONSTRUCTION_CURRENT.md`. The original
48-arm study is complete, with adverse external controls. Current prototypes
address validity-conditioned sampling and reversible topology/geometry moves.
The manuscript below remains the previous candidate's audited development draft;
it must not be presented as an already validated description of the new route.

Working title: **Refining Molecular Generators with Exact Entropy Changes**.
The primary manuscript now describes the implemented method, its complete
volume/entropy derivations and the actual controlled evidence. It is a development
draft, not an ICLR-ready paper. Latest PDF:
runs/verification/paper_20260911_physical_final/main.pdf (7 main pages,9-page cap).
Historical scalar-ordering audit: paper/legacy_audit.tex and preserved sections/PDFs.

The first full external comparison is adverse: full EACF improves the joint KL
bound and independent GFN2 strain substantially more than convex on condition0.
A14900-parameter compact control also has more-negative joint changes than our
19365-parameter marginal changes, but mixed independent geometry. Whole-allocation
costs are196/198 versus139/144 seconds; the older full-model20x ratio cannot
support a general efficiency claim. All saved parents replay and every geometry
failure is retained. See evidence/eacf_full_compact_comparison_v1.json,
evidence/eacf_geometry_comparison_v2.json and NEXT.md. These results weaken the
current method case; they do not complete an ICLR contribution.

The matched-query physical MALA/HMC comparison also favors the direct controls
in independent strain and cost on condition0. All four18048-query replicas and
their full stochastic traces pass audit;128 xTB attempts retain119 converged and
9 failures. The new manuscript includes these results and explicitly does not
assign MCMC endpoint KL, density or equilibrium status. See
evidence/parity_physical_controls_v1.json and notes/physical_control_decision_v1.md.

## New target/source review

The real oracle mirror audit finds .0533--.2262-eV defects, far above proper
rotation/permutation roundoff. The source GVP also has cross-product features;
do not assume source O(3) invariance from the adapter's symmetry. The old raw-target
training jobs were cancelled before execution. The inversion-mixture source and
energy-averaged target now pass their prescribed checks. Original raw-target KL values are
not relabelled as new-target results. See notes/parity_refinement_candidate.md.
The new main text now records this limitation; physical projection preserves the small N8 gain, while
broader performance remains pending.

Current AI/physics contribution boundaries: `NOVELTY_POSITION_CURRENT.md`.
The newer production prefix has39 qualified arms; results are mixed across
conditions. See evidence/parity_production_prefix_v5.json and the six-condition
geometry prefix before using the historical N8-only checkpoint below.

## Scientific question

Can a fixed implicit molecular generator be refined toward a specified physical
energy distribution using a tractable symmetry-preserving neural transport,
without estimating its source score or absolute density? Geometry, entropy,
probability mass and total compute remain separate evidence requirements.

The candidate decomposes Cartesian coordinates into element-group internal
coordinates and element-centroid pairs. Invariant neural contexts condition
nonlinear convex point maps. Exact COM volume uses3-by-3 factorizations including
the centering correction. The source stays frozen; the unknown base entropy and
target normalizer cancel in the relative-KL objective. These identities and
basic equivariant/convex flow principles are established mathematics. Potential
novelty is the useful molecular architecture and learning intervention, which
still requires broad results and strong external comparison.

## Evidence available

One8-atom condition, two matched training streams, shared2048-row confirmation:
convex-minus-typed DeltaKL=-.04892+/-.01732 and-.03882+/-.01642 nat; versus the
same-neural-context affine tangent control=-.01332+/-.00252 and-.01259+/-.00215.
SEM is over rows conditional on a model; do not pool streams as4096 independent
samples. Both control families and the candidate replay from frozen checkpoints.
Path ESS remains approximately1/2048; calibrated sampling is not established.
Prior geometry panels converge32/32 per arm, with small median strain changes;
this is not full chemical validity or mode coverage.

Corrected24-arm engineering passes with2304 raw queries, and294 repository tests
pass. The new comparison script also replays all24 saved corrected cases. The
earlier raw engineering panel had1536 queries and remains separate evidence.
Full8-condition/3-family/2-stream corrected production is submitted as
45914819/45914826, with outputs still incomplete. Reserved722 conditions remain
untouched. No HMC or external
coupling-flow superiority is established. The source timeout, failed score actors,
CNF, work, covariance, empirical-CFM and auxiliary experiments remain retained.

## Conditions for a credible paper

Target: labelled, unweighted COM space; fixed composition/charge/spin; conservative
eSEN plus .05 sum||x||^2, physical kT=.025851999786435 eV. This is a confined ML
potential, not an unconfined experimental or direct-DFT ensemble. Flow time is
not MD time. Energy histograms include density of states.

The broader source is a separately trained displacement FM64 map plus .025-A COM
Gaussian noise. Its pretrained1-eV neural input is retained as a feature, not a
thermal label inferred from OMol. No source importance density is evaluated.
Conditional3D configurations do not qualify unconditional composition generation.

External EACF source is pinned and its joint/marginal objective distinction is
recorded in notes/eacf_comparison_contract.md. Its implementation is not yet a
qualified baseline here; isolated dependencies/core imports pass, but model and
energy-bridge qualification remain outstanding. See CLAUDE_HANDOFF.md at the
repository root for the execution runbook. The final contribution needs meaningful replicated
cross-condition gains, full geometry/diversity/failure reporting, credible
independent distribution checks, matched total compute and reproducibility.
The main text now has space for that evidence; pending results must not be filled
with optimistic or inferred numbers. The original full ICLR objective remains open.
