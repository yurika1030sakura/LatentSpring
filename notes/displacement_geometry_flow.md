# Residual velocity head: a separate conditional flow protocol

September 9, 2026. This is an architectural development experiment, not a
reinterpretation of archived model weights as an already trained velocity.

Write the raw coordinate output of the reused FlowMol backbone as D_theta(x,t).
For `position_parameterization=displacement`, define the centered field as

    v_theta(x,t) = P[D_theta(Px,t) - Px].

The network has residual coordinate updates internally; this adapter uses their
net displacement directly as velocity. It does not multiply by alpha'/(1-alpha).
The existing `endpoint` mode retains that conversion and its own checkpoints.
All modes disable history/self-conditioning and use identical composition
clamping in training, sampling and density evaluation. The dependency is not
modified. COM-free Gaussian prior, frozen composition factor and bond-free
training are unchanged.

For tau=alpha(t)/alpha(T), draw independent X0 and X1, set
Xt=X0+tau*(X1-X0), and fit the raw head to

    D* = Xt + alpha'(t)/alpha(T) * (X1-X0).

Raw head MSE is now exactly velocity MSE; no time-dependent rescaling is hidden
in the loss. This allows a full data endpoint T=1. The direct velocity adapter
has no endpoint denominator, although the learned field can still have large
Jacobians or other numerical difficulties. It is not a convergence guarantee.

## Bounded first comparison (declared before results)

Two 10,000-step runs start at the same original 30,000-step FM checkpoint, with
seed 9004, identical full-OMol training order and random prior/time draws,
learning rate 2e-5, one molecule per batch, max_atoms=200, T=1 and no energy or
force training. Both use displacement mode. One keeps the original backbone
geometry; the other uses the separately declared smooth normalization rho=0.1
Angstrom. The two runs are not additional independent seeds.

Each job then evaluates the same eight composition-disjoint development
conditions and four Gaussian priors per condition at 64 and 128 sampling steps,
followed by independent GFN2-xTB evaluation, retaining all failures. Finally it
checks the predeclared sigma=[0.03,0.06] perturbation panel at 64/128/256 density
steps using eight independent replicas fixed over time and resolution. All
panels and initializations are shared with earlier development cases. This
reuse is development, not blind testing.

The existing 0.1-nat maximum mean-centered 128-to-256 difference is retained as
a necessary local numerical gate. Passing it does not certify exact density:
it must be followed by a stricter-resolution/adaptive check, probe uncertainty,
and full-neighbourhood assessment. Failure must not be repaired by dropping
parents after seeing results. All older endpoint results remain reported.
