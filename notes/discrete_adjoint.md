# Memory-bounded derivative of the discretised CNF objective

The 128-step checkpointed pilot completed all four updates but used 19.03 GiB
and roughly 150 seconds per step on a 20-GB MIG device. This motivates a
separate, tested discrete adjoint backend, not a frozen-trajectory shortcut.

Write each reverse-time numerical step as x_{j+1}=F_j(x_j,theta), with trace
quadrature d_{r,j}(x_j,theta) for replica r, and

    ell_r = log p0(x_S) - sum_j h d_{r,j}(x_j,theta).

Given upstream weights w_r=dL/dell_r, initialize the state adjoint to the
Gaussian prior derivative, summed over replicas. In reverse step order,

    a_j = (dF_j/dx_j)^T a_{j+1} - h sum_r w_r (dd_{r,j}/dx_j),
    dL/dtheta += (dF_j/dtheta)^T a_{j+1}
                - h sum_r w_r (dd_{r,j}/dtheta).

The implementation stores every forward coordinate state and probe tensor,
then reconstructs the local differentiable computation at that exact stored
state. It does not numerically invert the forward trajectory in backward,
and the state adjoint carries all trajectory and prior dependencies. It is
the chain rule for the discretised objective, not an assertion of unbiased
continuous-ODE gradients.

Working memory consists of stored coordinate/probe states plus one step's
network/derivative graph and accumulated parameter gradients. Standard
autograd through the entire solve remains available as the correctness
reference. The custom backward intentionally supports first parameter
derivatives only; higher parameter derivatives and coordinate/temperature
input derivatives are unsupported. Neural-network second derivatives needed
for the first parameter derivative of divergence are computed locally.

Tests compare values and all used parameter gradients against ordinary
autograd on the actual CTMC network. Nonlinear, state-dependent divergence,
zero-divergence shifts, exact traces, independent replicas, Gaussian prior
contributions and dedicated RNG streams are covered. Finite differences
independently check the parameter gradient. Probe draws occur once during
forward and are reused during backward; no fresh stochastic objective appears
on recomputation. A GPU rerun with the same four batches and seed measures
actual memory, time, objective and weight agreement.
