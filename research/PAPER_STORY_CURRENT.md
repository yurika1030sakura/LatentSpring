# Current paper direction — September10,2026

Working title: **Refining Molecular Generators with Exact Entropy Changes**.
The primary manuscript now describes the implemented method, its complete
volume/entropy derivations and the actual controlled evidence. It is a development
draft, not an ICLR-ready paper. Latest PDF:
runs/verification/paper_20260910_refinement_v4/main.pdf (5 main pages,9-page cap).
Historical scalar-ordering audit: paper/legacy_audit.tex and preserved sections/PDFs.

## New target/source review

The real oracle mirror audit finds .0533--.2262-eV defects, far above proper
rotation/permutation roundoff. The source GVP also has cross-product features;
do not assume source O(3) invariance from the adapter's symmetry. Full raw-target
training is held before execution. A versioned inversion-mixture source and
energy-averaged target are being qualified. Original raw-target KL values are
not relabelled as new-target results. See notes/parity_refinement_candidate.md.
The new main text now records this limitation; physical projection outcomes and
broader performance remain pending.

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

All24 broader engineering cases pass with1536 queries, and289 repository tests
pass. Full8-condition/3-family/2-stream production is submitted, with outputs
still incomplete. Reserved722 conditions remain untouched. No HMC or external
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
qualified baseline here. The final contribution needs meaningful replicated
cross-condition gains, full geometry/diversity/failure reporting, credible
independent distribution checks, matched total compute and reproducibility.
The main text now has space for that evidence; pending results must not be filled
with optimistic or inferred numbers. The original full ICLR objective remains open.
