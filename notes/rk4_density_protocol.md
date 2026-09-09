# Fourth-order integration as a numerical development control

The separately trained residual-velocity field permits finite evaluations at
both endpoints. Add classical RK4 to the joint state/divergence ODE, while
keeping the midpoint implementation available. This is a standard numerical
method, not a new flow or estimator. It changes neither model weights nor the
ideal continuous flow, but its finite map and quadrature must be identified.

For a reverse step of length h, the stages evaluate (x,t), (x-h k1/2,t-h/2),
(x-h k2/2,t-h/2), and (x-h k3,t-h). State and divergence use weights (1,2,2,1)/6.
The sampling implementation uses the same stages in the forward direction.
Endpoint-head mode still requires T<1 for RK4; midpoint's interior-stage
endpoint handling must not be copied to a method that evaluates t=1.

Each trace replica uses the same probe across the four stages within a step;
replicas remain independent. Under training, probes refresh at the next step.
The resulting conditional trace remains unbiased, but variance formulas must
retain within-step cross-stage covariance. Replace the single-stage Jacobian
by the weighted sum of stage Jacobians when applying the quadratic-form
identity. The discrete adjoint caches and reuses those probes and differentiates
all four stage states and the prior.

Analytic tests independently check fourth-order state and density convergence
for a time-dependent linear field. Nonlinear divergence and the actual FlowMol
network compare ordinary autograd, checkpointing and discrete-adjoint parameter
gradients; finite differences provide an additional gradient check.

## First molecular check

Use the two completed displacement-head checkpoints and the same eight
composition-disjoint local parent groups (sigmas 0.03, 0.06 Angstrom), fixed
geometry and eight fixed-over-time Rademacher replicas. Test RK4 at 16, 32,
64 steps and retain every parent. The 0.1-nat centered-mean difference remains
a necessary numerical threshold, followed by higher resolution or independent
adaptive confirmation. It is not sufficient to average away large per-replica
errors.

Step counts are NOT equal-compute comparisons: midpoint uses two field
calls and one divergence evaluation per step; RK4 uses four field calls and
four divergence evaluations. Report wall time and both counts. The three RK4
resolutions above sum to 448 field calls and 448 divergence evaluations per
parent, compared with midpoint 64/128/256's 896 field calls and 448 divergence
evaluations. Each divergence evaluation uses eight vector-Jacobian products.
Do not run large physics training until its selected density discretization
has passed the necessary numerical checks.
