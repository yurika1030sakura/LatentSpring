# Inversion-consistent source and energy refinement

Status: foundational repair candidate after a real-potential audit. Do not unhold
the old broad training jobs until a new source/target protocol is qualified.
The old raw-potential results remain valid statements about their original
declared experiments; they are not silently relabelled as new-target results.

## Measured issue and a necessary distinction

Audit45899596 evaluated4 fixed geometries in each of8 conditions(256 queries).
Raw eSEN mirror energy differences reach .0533--.2262 eV per condition. Proper
rotation errors are below5.73e-6 eV; joint atom/element permutation errors below
3.58e-6 eV. The discrepancy is therefore not explained by ordinary rotation/
permutation roundoff. The h=.001-A force-direction checks pass their stated
tolerance; h=.0003 fails on conditions3,6,7 and needs a broader FD ladder.

The adapter is O(3)-equivariant, but this alone does NOT make its output law
O(3)-invariant. The actual FlowMol source config has n_cp_feats=4; upstream GVP
mixes polar and cross-product vectors. It is SO(3), not guaranteed O(3).
Do not assert that the original source distribution is reflection invariant.
Non-equivariance of a pointwise source map also does not by itself prove a
distributional symmetry defect. Existing numerical claims must keep this scope.

For an inversion-equivariant bijection T and inversion R=-I,
T#(R#q)=R#(T#q). Consequently any invariant-under-bijection f-divergence between
q and its inverted law is unchanged by T. Such an adapter cannot manufacture
missing invariance in its source. This is standard push-forward reasoning,
not a new theorem claim.

## Explicit repair for the present composition-only target

Use q0_plus=(q0+R#q0)/2 and
E_plus(x)=[E_raw(x)+E_raw(-x)]/2. The physical target is proportional to
exp[-(E_plus(x)+.05 sum||x||^2)/kT], with the same kT=.025851999786435 eV.
This is a NEW target and a NEW source law, despite unchanged raw checkpoint.
Forces are F_plus(x)=[F_raw(x)-F_raw(-x)]/2. This sign is essential.
Group averaging restores inversion symmetry and retains conservative forces.
It removes the odd part of the model error for an inversion-even true energy;
it does not certify the remaining ML error or equilibrium coverage.

The current target conditions only on composition, charge and spin on the full
COM-free space. A target restricted to a specified absolute stereoisomer or a
fixed chiral external environment would require a different symmetry/domain.

At inference, apply one independent random sign to each generated parent and
retain that sign as lineage. For an inversion-equivariant T, flipping before
or after T gives the same law T#q0_plus. Duplicating both signs of every parent
is a correlated paired ensemble, not twice as many independent samples.

## Exact entropy scope

The deterministic transport identity compares T#q0_plus with q0_plus:
DeltaKL=E_q0_plus[(U_plus(Tx)-U_plus(x))/kT-logdet DT(x)].
The bracket is even, so its expectation can also be evaluated on original q0
parents. No source density or source-mixture entropy evaluator is needed.
The fixed source augmentation is NOT itself an invertible map whose entropy
change equals zero. Its separate KL decrease toward an even target is
KL(q0||q0_plus)=JS(q0,R#q0), lying between0 andlog2. The total change relative
to original q0 is therefore the transport change MINUS this nonnegative gain.
Do not label the transport-only number as an exact total change from raw q0.

An old explicit Gaussian-path work can be reused with a specified auxiliary
construction: draw a uniform sign a, output x=R^a y, and choose target auxiliary
(1/2)r_orig(path|R^a x). Its forward proposal has the same sign factor, so these
cancel. The new work is W_plus=W_raw+(E_plus(y)-E_raw(y))/kT. The original
source/reverse path need not be inversion invariant. After an equivariant
adapter, apply this same target correction to its old work. This establishes
a valid new weighted experiment, not retrospective calibration of the old one.

## Next bounded validation and training decision

1. Qualify exact even-energy/force algebra, non-invariant-source mixture entropy,
   explicit sign-augmented work and adapter inversion equivariance. Unit tests
   are implemented. Run real projected-potential symmetry and FD ladders.
2. Re-score the stored N8 shared2048-parent base and all six fixed adapters using
   inverted raw energies. Keep the old results; report new-target relative KL
   and weights separately. This tests whether the previous small gains survive
   the physical target repair. It does not establish broad usefulness.
3. Freeze a new broad training recipe and source sign handling before release.
   An unbiased unpaired estimator draws randomly inverted parents and queries
   raw E once each: expectation equals projected energy because output is even.
   A paired estimator uses half as many parents and evaluates both orientations
   at the same physical-query budget. Independent even-target evaluation must
   query both orientations. Do not silently double the training budget.
4. Only if measured odd-gradient variance motivates it, investigate a learned
   odd-energy control variate h(-x)=-h(x). Its mean and actor-gradient mean
   vanish under the invariant refined law, independently of critic accuracy.
   Poor fits can still increase variance, so coefficient fitting and matched
   independent validation are required. This is a prospective idea, not an
   implemented method or novelty result; it is not a proposal-score actor.

The parity issue alone has not been shown to explain ESS~1/2048. Preserve all
older failures and require new calibration evidence. Full ICLR goal remains open.
