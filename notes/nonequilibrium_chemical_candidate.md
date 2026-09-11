# Next bounded hypothesis: exchange with a reversible relaxation path

Status: the symmetric path and paired graph guide are now implemented and tested.
The v1/v2 physical pilots are complete; a frozen-selector composition is the next
comparison. No overall sampling or ICLR advantage is established. The work responds
to a measured failure: the action policy
learns rapid transitions for three parents but cannot repair the fourth; smaller
local steps improve local acceptance without crossing that parent's barrier.

Finite nonequilibrium proposals and pathwise Metropolis correction are established
methods, including [Nilmeier et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3215031/).
Read the [published correction](https://pmc.ncbi.nlm.nih.gov/articles/PMC3386054/)
before reusing its Brownian-integrator equations. The implementation below should
derive ratios directly from its actual Gaussian kernels; no new physics theorem
or generic NCMC novelty is claimed.

## Coordinate/path construction to implement

Implementation: `cfm_mol/escorted_exchange.py`, `chemical_path_guide.py`, and
`scripts/research/escorted_chemical_pilot.py`. The first pilot46043136 consumed
1384 raw queries per replica, accepted4/64 paths each, and left the difficult
parent unsupported at all16 proposed endpoints. Its full raw-output, probability,
path reversal and random-stream replay is in `escorted_chemical_audit_v1.json`.

The fixed graph-guided pilot46061289 consumes the same1384 queries per replica.
It improves supported endpoint counts from48/64 each to64/64 and63/64, but accepts
only4/64 each and still fails the difficult parent. The evidence is
`guided_escorted_chemical_audit_v2.json`. This separates a repaired proposal-validity
problem from the remaining acceptance/work problem.

The v3 protocol composes the existing trained selector with the unchanged guided
path, without new training. It conditions the selector on the exchange family;
both forward and reverse probabilities are normalized over exchange actions.
The fixed MALA stages retain their own kernels. Source preparation and5564
inherited training queries must count for each learned arm. This is a bounded
composition test, not renewed scale-up of the failed selector recipe.

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

## Fixed graph guide and its exact target correction

The pilot guide adds20 eV/A^2 bond springs around covalent-radius sums and
100 eV/A^2 soft nonbond exclusions below1.35 times those sums. It is a proposal
guide for the present single-bond graphs, not a physical force field. The
initial graph guides the pre-map segment; its deterministic terminal-edge swap
guides the post-map segment. A guided endpoint is eligible only if its perceived
bond graph equals that expected graph. The reverse then uses the same graph pair
in reverse order, including every reverse Gaussian mean.

Let G_old(x),G_new(y) denote the endpoint guide energies. The propagated smooth
densities are proportional to exp[-(U+G)/kT]. Therefore the path's smooth endpoint
log ratio must receive +(G_new(y)-G_old(x))/kT to recover the original physical
target. The guide is not added to the stationary distribution. This identity
is checked against the direct physical endpoint difference in the producer and
auditor. Independent autograd verifies all saved guide forces.

Tests include the full augmented-path Jacobian/involution, reversed Gaussian
path ratios, zero-length reduction to the terminal map, hard-endpoint support on
a known half-normal target, molecular guide force/symmetry checks and a known
normal target with paired guides. In the last test, omitting the endpoint guide
correction produces a wrong second moment; applying it preserves the target.
These checks establish implementation consistency, not molecular efficiency.
