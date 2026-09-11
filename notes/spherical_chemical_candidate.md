# Classical angular controls and the next directional proposal

Status: both angular controls are complete and fully audited. The force-informed
vMF kernel repairs the difficult parent in one of two replicas; the other still
fails. This is the first such transition in the present reconstructed controls.
Neither rotations nor vMF/MALA identities are claimed as AI novelty.

## Why change the coordinate update

The difficult generated development parent has two sulfur-bound hydrogens at a
38.02-degree angle. Local multiscale MALA makes accepted small moves but does not
resolve its constitutional-isomer barrier. Unrestricted nonequilibrium paths
mostly leave the hard support; fixed graph guidance repairs endpoint eligibility
but not path acceptance. Adding the frozen learned selector still leaves this
parent unresolved. The source, target and all negative results remain unchanged.

The uniform angular control alternates local MALA, a fixed-length terminal
rotation, a uniform chemical exchange, and local MALA. Gaussian quaternion
auxiliaries generate uniform SO(3) rotations about a passive anchor; conjugating
the quaternion is the reverse auxiliary. The map has unit COM volume and keeps
the terminal bond length fixed. The proposed perceived graph must remain equal
to the current graph. Accepted rotations are0/256 and5/256 in the two replicas;
the difficult parent does not change connectivity. Full replay is in
`research/evidence/angular_chemical_audit_v1.json`.

## Conditional angular density and the correct measure

Use labelled unweighted COM coordinates, dimension3(N-1). For a selected leaf i
and anchor k, a linear coordinate chart consists of passive relative positions
and v=x_i-x_k. Its Jacobian with respect to COM volume is constant. Writing
v=r u gives the measure r^2 dr dOmega(u). A terminal update holds r and every
passive relative coordinate fixed; the r^2 factors therefore cancel in the
forward/reverse conditional angular proposal ratio.

`cfm_mol/spherical_proposal.py` uses the S2 von Mises-Fisher density

    q(u;eta) = [||eta||/(4*pi*sinh(||eta||))] exp(eta dot u),

with the uniform limit at eta=0. This is NOT a full Cartesian endpoint density.
The sampler uses the exact inverse CDF for cos(theta) and an isotropic Gaussian
direction in the perpendicular plane. It does not assign a preferred spatial
axis at zero concentration. Tests independently integrate the density, check
the zero-parameter gradient and validate known vMF moments.

The physical control chooses a concentration LABEL c from[1,10,100]. With u the
current leaf direction, r its bond length and F the COM-projected physical force
(including the declared restraint), define

    s = r (I-u u^T) F_i/kT,
    eta = c u + 0.5 clip_norm(s,4 sqrt(c)).

The actual vMF concentration is ||eta||, not c. The reverse eta is recomputed
from the proposed geometry and its physical force using the SAME sampled label.
The MH ratio includes both vMF normalizers and the actual reverse leaf-selection
probability. Independent force labels and angular noise are saved. No oracle
Hessian is required. A known-target S2 test checks stationarity of this corrected
force-informed kernel.

Draw one angular proposal and reject it if the old chemical support or unchanged
graph condition fails. Retrying until validity would generally change the
state-dependent normalizer; the present force-based proposal does not implement
or assume such a cancellation. No old trajectory receives a new support label.

## Physical budget and learning gate

The frozen force-angular pilot retains the same four generated development starts
and256 microsteps per replica. The schedule is local/force-angular/exchange/local.
At most2056 raw queries per replica are allowed; each scored state includes both
inversion orientations. Source512 and development warm-up404 are inherited costs.
There is no learned controller or training cost in this physical control.

Job46065397 completed with1702/1886 new raw queries. Force-angular acceptance is
25/256 and43/256, compared with0/256 and5/256 for uniform rotations. Replica1
reaches the difficult parent's reference connectivity at step163 and finishes
with all four in that connectivity; replica0 still has only three. The difficult
parent's final energies are-25504.0037/-25506.7940 eV. Full physical-force,
conditional-density, proposal and random-stream replay is in
`research/evidence/force_angular_audit_v1.json`. One success is not a mixing or
robustness certificate. The next AI design is in `masked_angular_learning_design.md`.

Only after the directional proposal is audited should a learned angular model
be considered. It must beat these force-informed and uniform controls with
complete costs, preserve the conditional measure/reverse density and use only
generated training parents or other explicitly qualified training data. Improved
local acceptance alone is not equilibrium sampling, generalization or ICLR
readiness. The larger source-validity and metal/radical-support limitations remain.
