# BGFM: audited mathematical specification

Updated 2026-09-08. This replaces the missing derivation referenced by the
project guide. The April copy in laboratory storage is historical: its
population-optimum, Gaussian-coupling, support-normalisation and cross-batch
arguments are not a justification for the current experiments.

## Loss family and scope

Keep the bond-free FlowMol3 backbone, OMol25 data, and

`L = L_FM + lambda_1 L_force + lambda_2 L_energy (+ lambda_3 L_anchor)`.

FM learns the data distribution. Unless those data follow the chosen Gibbs
law, FM and exact energy matching generally prefer different densities. Large
weights alone do not prove that the energy residual is feasible or vanishes.
No current experiment establishes Boltzmann sampling across compositions.

## Score readout

For **independent** `X0 ~ N_H(0, sigma^2 I_H)` and `X1`, on the zero-centroid
subspace H, `Xt=(1-t)X0+tX1` gives

`score_t(x) = (t E[X1 | Xt=x] - x) / ((1-t)^2 sigma^2)`.

With the **population** velocity `v*(x,t)=E[X1-X0 | Xt=x]`,
`E[X1 | Xt=x]=x+(1-t)v*(x,t)`, hence

`score_t(x) = (t v*(x,t)-x) / ((1-t) sigma^2)`.

FlowMol3 returns an endpoint prediction `D_theta`, not this velocity. For its
position schedule alpha, convert with

`v_theta = alpha'(t)/(1-alpha(t)) * (D_theta(x,t)-x)`.

At linear alpha, passing D directly to the velocity readout scales the
intended endpoint-based score by `(1-t)` before clipping. A changed loss
weight cannot in general repair all three probe times and norm caps at once.
Neither readout is automatically the score of an arbitrary learned ODE.
Prior alignment conditional on X1 and steric retraction change the joint
path distribution; the independent-Gaussian derivation does not apply
unchanged. Even with a correct population velocity, the finite-t score is a
smoothed marginal score and is not generally `F(X1)/kT` or `F(Xt)/kT`.

## Which density is defined

Define the deterministic smooth clamped field
`v_H(x,t;c) = P v_theta(Px,t;c)`, with discrete endpoint labels held fixed,
no persistent self-conditioning history and no nonsmooth sampling projection.
For an interval `[0,T]` on which this field defines a diffeomorphic flow,

`log q_T(x) = log p0(phi_T^{-1}(x)) - integral_0^T div_H v_H(x_t,t) dt`.

`P` is the per-graph orthogonal zero-centroid projection; it must be INSIDE
the differentiated field. The prior has dimension `3(N-1)`. Centering only
the numerical input does not remove translation derivatives from the trace.
This clamped q_T is not the conditional density of the joint CTMC/retracted
sampler. Singular t=1 behaviour needs a separate boundary/convergence study.
The corrected implementation defaults to explicit T=0.95 and does not label
it an endpoint likelihood.

The old `log_density_via_flow` instead integrates the raw endpoint head,
does not center the differentiated field, and omits trajectory/prior
parameter derivatives. It is retained solely to reproduce archived results.
Do not call those stored numbers a validated likelihood of the molecular
generator, even after a finer legacy solve.

## Grouped energy residual

For a fixed parent m, `w_mk = log q_T(x_mk|c_m) + E_mk/kT` and
`L_energy = mean_m Var_k(w_mk)` (population divisor K in the code).
Within-parent constants cancel; geometry-dependent bias and trace noise do
not. Only the within-group form is supported for heterogeneous molecules.
A constant log density does not attain zero loss when energies vary. A
free constant head cannot by itself recover slope or relative basin mass.

For one group and nonconstant y=-E/kT, the OLS decomposition
`ell = a y + b + epsilon`, `Cov(epsilon,y)=0`, gives

`NRV = (a-1)^2 + Var(epsilon)/Var(y)`.

The best **unrestricted signed** scaling gives `1-r^2`. If only positive
rescaling is admissible (as for temperature), a nonpositive correlation has
infimum NRV=1 at zero scaling, not `1-r^2`.

Zero population residual variance identifies the Gibbs relation only on
the support of the reference measure, up to a constant per connected
component. Finite sampled groups constrain only their sampled points.
Disconnectedness gives separate offsets in an unrestricted measurable
function class; continuity or model restrictions may couple them additionally.
Global Gibbs claims also need finite partition integrals and a specified
domain: removing translation alone does not confine dissociating fragments.

## Differentiation and numerical validation

`clamped_density.py` converts the endpoint head with the real schedule,
centers states and fields, disables stochastic module behaviour temporarily,
and performs midpoint stages for both state and divergence. Training retains
the whole state/prior graph. It differentiates the discretised objective;
it is not an unbiased continuous-time likelihood-gradient estimator.
Even an unbiased stochastic trace gives a noisy log-density. Squaring a
centered noisy residual adds a covariance-dependent noise penalty in
expectation; an unbiased trace does not imply an unbiased energy variance.
Common random probes may reduce comparison noise but do not prove removal
of this penalty. Probe repeats or independent-trace products/exact traces
are needed to quantify it before interpreting a training effect.

Required checks: analytic COM-free Gaussian density, schedule conversion,
second-order convergence on smooth fields, finite-difference parameter
gradients with fixed probes, actual CTMC forward/backward, precision,
and checkpoint-level resolution/probe-repeat stability. The last checks
must precede any use of corrected density in a paper's molecular results.
