# Counterfactual conditional-density identification

This is a development method design and diagnostic, not a demonstrated new
sampler advantage. The linear algebra, vMF identities and exact MH correction
are established mathematics. No new physical law is claimed.

## What the completed evidence says

The original guide improves the72-parent one-composition endpoint potential at
matched raw-call cost, and independent GFN2 single points corroborate that
energetic ordering. Its frozen six-composition test does not improve the average:
learned minus site is+0.0978eV including training, with hierarchical95 percent
interval[-0.0161,0.2141]eV. At equal inference calls the mean is+0.0042eV.
The22/23-atom cases accept zero learned-component proposals in their common
200-step diagnostic prefixes. These results do not support broad transfer.

TRAINING-only angular probes now supply a more specific diagnostic. For32
post-exchange/warm contexts, hold every passive relative coordinate and the
selected radius fixed. Rotate the leaf by +/-0.15 radians in two orthogonal
tangent directions, and reserve +/-0.10-radian diagonal probes for checks.
All192 probes are supported. The32 centers and probes cost448 paired raw calls.
Full geometry/RNG replay and an independent normal-equation implementation pass,
with maximum fitted-parameter discrepancy5.92e-12.

## Identifiability and finite work

For a full-sphere vMF surrogate q(u|C) proportional to exp(eta(C) dot u),
the surface score is P(u) eta, where P(u)=I-u u^T. One observation at u cannot
identify eta's component parallel to u. Masking u from a neural context prevents
direct leakage but does not create that missing information for an otherwise
unique context.

For multiple directions sharing exactly the same context, least squares gives

    A = sum_j P(u_j), b = sum_j s_j, eta_hat = A^(-1) b.

For any v, v^T A v=sum_j ||P(u_j)v||^2. Thus A is positive definite when the
observed directions are not all parallel. The implementation independently
checks this rank condition. For a truly linear angular log potential, finite
work also satisfies -(U(u_j)-U(u_0))/kT=eta dot (u_j-u_0). The withheld work
readout checks this approximation without entering the force-based fit.

These statements do not identify an arbitrary nonlinear angular potential or
the normalizer of a hard-support conditional target. The fitted full-sphere vMF
is a proposal surrogate; support rejection and complete reverse densities still
belong in MH. Local force/work agreement is not equilibrium sampling evidence.

## Exact known restraint and local representation

In the anchor-relative chart, set the moved leaf to r*u and let S be the sum
of the passive relative positions. Centering all N atoms gives

    sum |x_COM|^2 = constant + r^2*(1-1/N) - 2*r*S dot u/N.

The harmonic restraint therefore contributes eta_gamma=gamma*r*S/(N*kT)
to the angular natural parameter. Its sign and COM factor pass an independent
full-coordinate derivative test. `LocalSiteGuide` separates this known global
term from a learned local residual. Smooth distance gates and local weighted
normalization remove global N/sqrt(N) pooling from that residual. A remote-
spectator tensor test checks this representation property; it is not a physical
molecular benchmark.

The fitted residual norm median is434.2, beyond the old64 bound. The full
intrinsic vMF concentration median is444.2 (range237.9--1034.8), and the intrinsic
teacher mode differs from the physical-site mode by a median5.98 degrees,
maximum31.21 degrees. On the withheld angular probes, freely fitted local vMF
force-score MSE is51.7, compared with1641.6 after capping the correction at64,
and1615.4/1668.1 for the frozen neural models. The fitted finite-work MAE is
0.270kT. These are training-context diagnostics and approximate local models.

## Mode/stiffness prototype and negative learnability result

`StiffnessSiteGuide` predicts an intrinsic unit mode and a separate positive
concentration (up to2048), then adds the known restraint parameter. Its
normalized density, high-concentration sampling and gradients are tested.
`vmf_teacher_kl` implements exact normalized teacher-to-student KL, with the
teacher mean direction A(kappa)*eta/kappa and A(kappa)=coth(kappa)-1/kappa.

The first pilot fits24 old TRAINING parents and withholds8. Both seeds reduce
fit KL strongly (to0.094/0.175), but withheld-parent KL worsens from4.079 to
13.077/6.258; withheld-angle force MSE also worsens. This is overfitting, not a
successful repair. The range/architecture were selected after inspecting the
full probe summaries, so these are internal diagnostics, not a blind test.
Do not release these pilot weights as a corrected molecular method.

The completed post hoc decomposition (`runs/stiffness_mode_diagnostic_v1`,
job 46148875) reproduces both checkpoint metrics and independently checks

    KL(vMF(k_t,mu_t) || vMF(k_s,mu_s))
      = KL(vMF(k_t,mu_t) || vMF(k_s,mu_t))
        + k_s A(k_t) (1 - mu_t dot mu_s).

On the eight withheld training parents, final width KL is only 0.0334/0.0681,
whereas direction KL is 13.0439/6.1902. Direction contributes over 98% of the
remaining error in each seed. Thus the student learns concentration but is too
confident in its direction on new parents. The physical-site control has local
teacher KL 1.997 at concentration 64 and 5.343 at 400 on these same parents:
merely increasing concentration amplifies direction error. These comparisons
are diagnostic local-surrogate KLs, not molecular trajectory outcomes.

Eight additional proposal-training compositions are frozen in
`research/evidence/proposal_training_panel_v1.json`, with four per size stratum
8--12 and 13--24. Fixed metadata ranks exclude both the old eight and evaluated
six compositions; energy values and generated structures cannot affect selection.
The initial geometry-only protocol permits 256 attempts per composition, zero
physical calls and no model fitting. All source failures must be retained before
any subsequent physical-data protocol is frozen. The generator now records the
explicit `fresh_training` stream. Selection/energy-independence tests and a full
regression replay of the previous six source audits passed (job 46149458).

## Next decisions

Physical controls with concentrations64 and400 are frozen and running across
the original72 parents and all96 transfer parents. They test whether a simple
width correction explains the apparent neural benefit. Both incremental reuse
and fully charged10108-call calibration scenarios must be shown for400; the64
control uses the pre-existing bound and no new physical calibration.

Before further sampling claims, compare controlled changes in representation,
training coverage and concentration regularization. Broader training must use
the old TRAINING streams/other authorized training inputs, not the six evaluated
transfer sources. More force examples do not automatically identify conditional
width; more concentration does not repair an inaccurate mean direction. Keep
the simpler sharpened physical controls, failed pilot and all source denominators.
