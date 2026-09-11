# Next bounded hypothesis: exchange with a reversible relaxation path

Status: a concrete design for implementation/testing, not a completed method or
new performance evidence. It responds to a measured failure: the action policy
learns rapid transitions for three parents but cannot repair the fourth; smaller
local steps improve local acceptance without crossing that parent's barrier.

Finite nonequilibrium proposals and pathwise Metropolis correction are established
methods, including [Nilmeier et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3215031/).
Read the [published correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC3386054/)
before reusing its Brownian-integrator equations. The implementation below should
derive ratios directly from its actual Gaussian kernels; no new physics theorem
or generic NCMC novelty is claimed.

## Coordinate/path construction to implement

Keep the old condition0 target and labelled COM measure. A valid initial x has
the ordinary uniform terminal-exchange action list A(x). Choose a from that list
before any propagation. With a fixed k, generate k unadjusted Langevin coordinate
steps, apply the existing invertible terminal map T_a, and generate another k
Langevin steps. Use the same step schedule in mirrored order on the two sides.
The intermediate coordinates may leave the hard chemical support. Score their
smooth even eSEN potential plus the existing COM restraint; do not pretend an
outside-support score is the logarithmic derivative of the hard target there.
Only the final candidate y must pass the old chemical support and admit the
inverse action a_bar. The map remains algebraically defined even if an intermediate
graph no longer has the originally selected chemical attachments.

Let g(v|u) be the actual Gaussian propagation density in dimension3(N-1), with
the fixed clipped-force drift used by the implementation. Reverse the saved
propagation path, swap a for a_bar, and apply T_a inverse at its middle. The
candidate log acceptance ratio is

    -(U(y)-U(x))/kT + log|det_H DT_a|
    + log p(a_bar|y) - log p(a|x)
    + sum_propagations [log g(u|v) - log g(v|u)].

For uniform action selection the policy correction is log(|A(x)|/|A(y)|).
The reverse path uses the actual reverse Gaussian means at every saved state.
In augmented coordinates, reversal permutes the path vertices and changes one
middle free vertex through T_a; its nontrivial Jacobian is the existing map's
intrinsic determinant. Derive and test this explicitly before the molecular run.
With k=0 the method must reduce to the already audited terminal exchange.

This is one MH decision for the whole path. Applying ordinary endpoint MH to
a map followed by uncontrolled relaxation would omit the path-density ratio and
bias the target. An asymmetric 'swap then relax' recipe also lacks the same
reverse protocol unless a reverse family is explicitly included.

## Implementation and evidence gates

1. Implement the primitive with all Gaussian noises, vertices, raw paired oracle
   energies/forces, log densities, action choices and the final acceptance uniform
   in its trace. Test augmented path reversal/Jacobian, reversal of the total
   log ratio, k=0 reduction and a known-target stationary numerical control.
2. Use the exact same four generated development starts, plus the existing
   uniform and multiscale controls. Start with a small fixed physical budget;
   score BOTH inversion orientations and retain every rejected endpoint/path.
   Do not use evaluation/QC geometries to train any path controller.
3. Count every intermediate force query. A gain per attempted path can vanish
   per physical query. If a node/force evaluation fails, preserve its prefix and
   requested/acknowledged counts; do not treat it as a zero-cost rejected move.
4. If the physical path resolves the measured geometry barrier at useful cost,
   then test whether learned action/path selection adds value over that stronger
   physical control, with source preparation and training included. Generic
   NCMC, MH, Gaussian densities and Jarzynski motivation are prior art.

The initial FM distribution is not equilibrium and has no qualified density.
Do not infer Jarzynski normalizers, source importance weights or calibrated
finite-time endpoints from these paths. The valid claim would be target invariance
of an audited Markov kernel, followed by separate mixing/performance evidence.

Other recorded gaps remain: terminal moves cover only two of eight current
compositions; neutral-radical graph assignment is not yet a production support;
metal chemistry remains unsupported. More general pendant-fragment moves can be
investigated after the measured condition0 geometry barrier is understood, not
added as unvalidated capacity or an automatic novelty claim.
