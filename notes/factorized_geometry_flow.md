# Separating the composition prior from the geometry flow

The user authorised framework reconstruction on September 8. The next
experimental branch makes the sampler-density relationship explicit:

    m ~ pi_frozen(m),       X0 ~ N_COM(0,I),       X ~ q_theta,T(.|m).

The composition prior can be a frozen original FlowMol checkpoint. Its
sampled coordinates are discarded; only atom types, charges and atom count
condition the new position flow. Thus the full law is pi_frozen(m) q_theta,T(x|m).
We claim a tractable conditional geometry density, not a tractable pi_frozen.
The fine-tuned geometry checkpoint MUST NOT replace the frozen composition
checkpoint: shared backbone updates also change its discrete predictions.

The geometry flow uses the same memoryless COM projection, endpoint-to-velocity
conversion and terminal time in training, sampling and density evaluation.
For alpha(t), define tau(t)=alpha(t)/alpha(T) and an independent Gaussian path
Xt=X0+tau(t)(X1-X0). The head target is

    D* = X0 + (X1-X0)/alpha(T).

Indeed alpha'(t)/(1-alpha(t)) (D*-Xt) = alpha'(t)/alpha(T)(X1-X0).
Projecting the head onto COM-free coordinates and regressing D* therefore
gives a positive time-weighted CFM objective for data at T. Merely using X1 as
the head target while claiming q_0.8 ends at the data would be incorrect.

The position branch replaces joint CTMC FM during its fine-tuning; it does not
add two incompatible conditional path targets to the same field. It reuses the
existing network without modifying the shared FlowMol installation. Force
weight is initially zero. The loss remains FM plus optional grouped energy.
No alignment, retraction, bond labels or history self-conditioning is used in
this continuous branch. max_atoms=200 and OMol25 remain unchanged.

This is an experimental architectural factorisation, not a completed result.
T=0.8 is a predeclared development setting to bound the head conversion away
from its t=1 singularity. It still requires solver convergence, a sufficiently
trained CFM baseline, matched energy controls and independent sampling metrics.
Energy labels' charge/spin conventions and finite-partition assumptions remain
limitations; factorisation does not establish global Boltzmann populations.
