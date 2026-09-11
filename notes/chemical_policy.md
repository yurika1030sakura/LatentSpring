# Learned action selection on a declared molecular support

This is a development candidate, not an established ICLR contribution. The
learned object is the probability of a move family and a reversible terminal
exchange. The coordinate maps and local MALA proposal remain fixed. Flow matching
supplies initial coordinates; its unknown endpoint density is not used in MH.

## Target and state

Fix labelled atomic numbers, total charge and spin multiplicity. Coordinates x
lie in the unweighted COM subspace H with dimension 3(N-1). Let D be the declared
connected/no-overlap domain intersected with successful deterministic RDKit bond
assignment and sanitization at the requested charge. The perceived graph G(x)
is an input to the policy. It is neither an OMol25 bond label nor a quantum
validity certificate. Spin remains a separate energy-model condition.

The target, assuming a finite partition function, is proportional to

    1_D(x) exp[-U(x)/kT],
    U(x) = (E_raw(x) + E_raw(-x))/2 + kappa ||x||^2/2.

The present fixed composition can span different constitutional isomers. It is
not a fixed-identity conformer target. The two raw energy and force evaluations
are required for every supported scored proposal; the conservative even force
is (F_raw(x)-F_raw(-x))/2. Neither individual chains nor energy reductions provide
an equilibrium certificate or a finite-time importance density.

## Augmented actions and detailed balance

The policy returns p_theta(0|x) for local MALA and p_theta(a|x) for each eligible
terminal exchange a. Its invariant message-passing network uses atom identities,
distances, perceived bond orders and electronic/temperature conditions. The
local family probability lies in [0.1,0.9] when exchanges exist. The conditional
exchange probabilities contain a 0.1 uniform defensive mixture. With no eligible
exchanges, local probability is one. All probabilities are normalized over the
actual action list; unsupported destinations still retain proposal probability
and are rejected.

An exchange a=(i,j,k,l) maps terminal atoms i,j from passive anchors k,l to the
opposite sites, rescaling their relative vectors by covalent-radius ratios.
Its inverse is a_bar=(i,j,l,k), and the absolute Jacobian on H is

    J_a = [(r_i+r_l)/(r_j+r_l) * (r_j+r_k)/(r_i+r_k)]^3.

Rejection is mandatory when the proposed state is outside D or a_bar is absent
from its newly perceived graph. Otherwise the acceptance log ratio is

    b_a(x) + log p_theta(a_bar|T_a x) - log p_theta(a|x),
    b_a(x) = -(U(T_a x)-U(x))/kT + log J_a.

The incoming and outgoing augmented probability flows agree after changing
variables through T_a. This is ordinary reversible-map Metropolis correction,
not a new detailed-balance theorem. Multiple possible actions cause no problem:
each action has its own inverse pairing and the reversible components sum.

For local MALA the mean is x + sigma^2 clip(score(x))/2 in an orthonormal basis
of H. Its b_0 includes the reverse-minus-forward Gaussian log density. The
state-dependent *local family* probabilities must also enter the MH ratio.
Dropping them biases the target even if the conditional local proposal is
correct. A fixed local-only warm-up instead uses local-family probability one.

## What the offline objective estimates

Training x states come only from eligible generated TRAINING parents followed
by a fixed chemical-supported local warm-up. Every eligible discrete exchange
is enumerated. Four independent local proposals per x approximate its local
proposal integral. The proposals, graph checks, physical scores and utilities
are fixed with respect to theta. No evaluation or QC coordinates enter training.

For each retained table edge, define the symmetric utility

    R(x,y) = 1[connectivity(x) != connectivity(y)]
             + min(1, ((U(y)-U(x))/kT)^2).

Its accepted action mass is

    min[p_theta(a|x), exp(b_a(x)) p_theta(a_bar|y)].

We maximize the empirical source average of this mass times R, summing exchange
actions and averaging the four local draws. Automatic differentiation includes
both forward and reverse policy probabilities. There is no categorical
pathwise derivative through an action sample. The hard support is fixed for
the table, so its invalid edges have zero accepted mass; this argument would
not justify differentiating a learned coordinate map using the same table.

The source distribution is not assumed stationary. This finite-data movement
objective does not bound the true spectral gap, certify mixing or minimize an
established equilibrium-distribution discrepancy. Improving it may fail the
sampling experiment. Positive-mass acceptance ties retain the usual minimum
function differentiability qualification.

## Experimental gate

The v1 table uses all 34 eligible parents among 4096 generated training samples;
four eligible parents among 512 generated development samples are separate.
Training preparation cost is 5564 raw queries; development warm-up is 404.
The source preparation already used 4096/512 raw queries respectively. These
are actual inherited costs, not newly rerun calls. Source neural generation,
offline training and chain runtime remain distinct from raw-query counts.

The v2 evaluation protocol corrects the full-cost uniform control before any
evaluation outcomes. Each baseline runs up to 1482 steps; the learned policy
runs 256. Compare the baseline checkpoint at or below the learned arm's actual
total cost, and show the equal-256-step sampling prefix separately. Uniform
baselines use both 0.5 and 0.1 local-family probabilities. The latter was added
after training saturated the 0.1 floor, before evaluation, so action selection
cannot receive credit merely for changing the move frequency.

The six arms use two seeds and four development starts for one composition.
All proposals, invalid reasons, raw paired oracle outputs, cached states,
selection probabilities and acceptance uniforms are traceable. A replay of
saved oracle outputs checks implementation consistency; it is not an independent
physical-energy validation. Sampling/generalization advantages and AI novelty
remain open until the comparisons and independent molecular tests support them.
