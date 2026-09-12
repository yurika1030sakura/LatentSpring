# Next bounded test: normalized joint graph and geometry proposal

Status: design only. The masked tensor guide now generates valid angular
candidates efficiently and accepts75/256 and58/256 updates, but neither replica
crosses the difficult graph barrier. Further changes to angular acceptance alone
do not address this outcome. Preserve both envelope versions and their controls.

## Concrete move

Select unlike single-bonded terminal atoms i,j at DISTINCT passive anchors k,l.
Use the existing deterministic graph edge exchange to define the desired graph
G': i attaches to l and j to k. Propose BOTH new bond lengths and BOTH new
directions, then recenter COM. This replaces the old rule that transports the
source bond-length strain and ligand directions to the new chemical environment.
Require the final perceived bond matrix to equal G' and the inverse graph action
to be eligible; otherwise retain the old state without a physical query.

Begin with a fixed log-normal radial prior around each new covalent-radius sum,
with a prospectively frozen width. Its log-radius density is explicit. The
Cartesian conditional density for each moved root contains

    log Normal(log r; mu, sigma^2) - 3 log r + log q_angle(u | context,r,G').

The r^3 factor includes r^2 from spherical volume and r from d(log r). A linear
chart using passive relative coordinates and the two moved Cartesian vectors
has a constant COM-volume factor, which cancels. Do NOT reuse the Jacobian of
the old deterministic radius swap for these newly sampled radii.

## Normalized directional decoders and fair controls

Compare uniform directions, a fixed coordination-site prior, and the already
trained vector/tensor guides. A useful physical site prior points away from the
sum of other bonded-neighbor unit vectors; retain it as a baseline, not AI novelty.
Use actual normalized vMF or two-vMF-mixture densities for the learned decoder.
The current `angular_envelope` mixture can be used AS the normalized proposal,
without the additional rejection-to-Fisher-Bingham-score step. Its log density is

    eta.u + logcosh(gap * axis.u) - log_base_partition.

It is then a guide-derived proposal, not the normalized density of the original
linear-quadratic score. The old same-context score-ratio cancellation DOES NOT
apply across changed graphs, radii or passive contexts.

## Autoregressive construction and reverse evaluation

Draw both radii first. A deterministic template can place the new roots at
their new anchors using transported source directions, with the new radii.
Choose either root order with probability1/2; retaining only atom-index order
would break the intended permutation symmetry. Mask the first root, predict
its normalized angular distribution from this template and G', draw its new
direction, then do the second root using the first root's generated position.

The two conditional angular densities form a normalized autoregressive joint
proposal. Reverse evaluation must reconstruct this SAME generative procedure
from y with the inverse action, target radii from x, and the SAME augmented
root order. Insert each observed old direction as that step's forced output.
Do not evaluate both densities using only the completed endpoint: the first
conditional saw the template's second-root position, not its eventual draw.

The MH log ratio is the original physical target difference plus the complete
reverse-minus-forward joint coordinate log density and the actual reverse/forward
action-selection ratio. The order probabilities cancel. Failed endpoint graph
checks are self-transitions. No source likelihood or unknown angular-support
normalizer is introduced because directions are drawn once from normalized
full-sphere distributions and support is checked only at the final endpoint.

## Tests before molecular evaluation

1. Independently derive and test the conditional COM chart and r^3 measure,
   forward/reverse density evaluation and the graph-action inverse.
2. Check the mixture over both root orders under atom permutations and orthogonal
   transformations. Preserve the graph/electronic condition in every record.
3. Test a known full-coordinate target with nontrivial radial proposals; an
   angular-only stationarity test does not cover the new radial factors.
4. Retain all radii, directions, source templates, conditional model outputs,
   normalized log densities, validity failures and random streams for replay.
5. Use the same four generated development starts, fixed local/force-angular
   stages and a prospectively bounded query budget. Include a deterministic
   exchange control restricted to the SAME different-anchor action list, so
   removing neutral same-anchor swaps cannot receive credit as learned geometry.
6. Do not retrain on development coordinates or enlarge the neural model before
   this controlled test. Count inherited training/source costs for neural arms.
   A positive result still needs independent molecular conditions and distribution
   validation before an ICLR claim.
