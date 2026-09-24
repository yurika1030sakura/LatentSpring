# Frozen-parent geometry context

This independent architecture experiment starts from the original fitted FM
parents, not a model selected using recovery-pilot outcomes. It can run alongside
the continuation experiment and uses its fixed development compositions, draws,
and frozen-parent outputs as shared controls. Both studies are exploratory.

The adapter augments the existing provisional-distance self-conditioning channel.
A soft geometric neighborhood supplies element context, coordination summaries,
the mean neighbor direction, and the3x3 second moment of neighbor directions.
Scalar contractions and determinants encode local angular shape and departure
from a plane without supplying a bond graph or requiring a particular ring to
be flat. The features are invariant under rotation/reflection and permutation.
A zero-initialized bounded MLP adds a learned correction to the distance channel;
the pretrained network's predictions are preserved at initialization.

The moment adapter and radial control each have7,489 trainable parameters.
The radial control replaces the six angular/moment channels with six radial
basis functions; network dimensions, initial weights and training schedule match.
The complete pretrained backbone and physical-correction head remain fixed.
Both variants use the same recovery-local objective and original CFM replay,
4,000 updates per fit, and two fitted parents. The new context is used during
both passes of the unchanged128-call midpoint sampler. The physical head remains
at strength4; no hydrogen readout, sampling-time energy call, or optimization.

The additional budget is16,000 small-adapter optimizer updates,1,024,000 backbone
training-example forwards through a frozen backbone,1,536 new generated outputs
and fixed-coordinate GFN2 attempts. The768 frozen controls are reused, not
regenerated or counted twice. One3-hour GPU allocation is requested; together
with the recovery pipeline, requested GPU time remains below the first
experiment's original18-GPU-hour ceiling.

This tests the utility of local multi-atom geometry information. Moment features,
equivariant networks and zero-initialized adapters are established constructions;
an initialized module and passing tests are not generation-quality evidence.
The released controls distinguish angular information from added network size.
Five tests cover equal-radius planar/tetrahedral neighborhoods, zero initialization,
symmetries, gradients through the actual frozen EGNN, radial-control parameter
matching, and coincident-atom numerical behavior.

The running continuation experiment uses its original immutable source and is
unaffected by this independent module. Neither study changes the manuscript
until actual raw-output results are available and their scope is established.
