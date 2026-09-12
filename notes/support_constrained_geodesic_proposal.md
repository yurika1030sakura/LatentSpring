# Support-constrained conditional proposals

This is an implemented single-root prototype, not an established ICLR method.
The original generator remains flow matching. Geodesic walks and spherical slice
sampling are prior art: [Habeck et al., JMLR 2025](https://jmlr.org/papers/v26/23-1158.html).
The intended contribution requires a useful learned molecular proposal and its
comparison against strong physical versions of this same construction.

## Evidence motivating the reconstruction

The new multi-composition data contain 438 fixed contexts with independently
verified force fits and finite-work checks, costing 5,536 new raw calls.
Local fitted work MAE is about 0.425 kT, compared with 4.101 kT for the physical
site template. These are local training diagnostics, not a neural result.

Forty-seven local fits have eta dot u_center < 0. Full-sphere sampling from these
fits yields only 16 structurally supported proposals out of 376, whereas the
physical site prior yields 375/376. In the remaining contexts the fitted proposals
are mostly supported (3091/3128). Thus a locally accurate fit must not automatically
be treated as the true full-sphere conditional density. Do not run the planned
unrestricted teacher-KL distillation on all these labels as though that assumption
had passed. Keep every context and every failed proposal.

## Geometry and the implemented probability law

For one terminal atom with anchor-relative vector r*u, hold r, the desired graph
and all passive relative coordinates fixed. Our current RDKit graph contract uses
`useVdw=True, covFactor=1.25, embedChiral=False`. The pinned 2025.03.2 source and
[RDKit API](https://www.rdkit.org/docs/cppapi/namespaceRDKit.html) verify this
connectivity convention. A nonbonded passive atom at w requires

    u dot w <= (r^2 + |w|^2 - d_min^2)/(2*r),
    d_min = 1.25*(R_root + R_passive) + margin.

The root-anchor radius is separately checked against the bond and overlap bounds.
Passive geometry must already be qualified; the final bond-order/charge validator
is still applied. These statements concern algorithmic structural support, not
quantum stability or electronic-state certification.

Choose v uniformly on the unit tangent circle at a reference direction u0 and set

    u(theta) = cos(theta)*u0 + sin(theta)*v,  0 <= theta < 2*pi.

Every distance constraint becomes a*cos(theta)+b*sin(theta)<=c. Their intersection
is a finite union of intervals, computed analytically. A single empty-circle draw
is a failed proposal; it is never resampled until success.

On each admissible segment [a,b], the implementation linearly interpolates endpoint
scores: g(theta)=g0+d*(theta-a)/(b-a). Its mass is exactly

    (b-a)*exp(g0)*exprel(d),  exprel(d)=(exp(d)-1)/d.

The categorical segment choice and exponential inverse CDF use this same law.
Stable small- and large-slope formulas are tested. The proposal is the normalized
interpolant, not an unevaluated exact Boltzmann conditional or a continuous vMF
whose normalizer has been omitted.

For an observed direction u away from +/-u0, put theta=atan2(s,c),
c=u dot u0, s=|u-c*u0| and v=(u-c*u0)/s. There are TWO oriented-circle preimages.
The density per unit sphere area is

    q(u | u0,C) = [p(theta | u0,v,C) + p(2*pi-theta | u0,-v,C)]
                  / [2*pi*|sin(theta)|].

Both terms are evaluated, including each actual interval normalizer. The
spherical Jacobian and the two-preimage factor pass independent checks. Excluded
numerical charts and empty circles contribute self-transition probability; no
unknown success-conditioning normalizer is inserted or dropped.

## A compatible learning objective

The local force/work fits can serve as ENERGY-SCORE surrogates on valid arcs;
they must not be promoted to unconstrained probability teachers. For teacher and
student on the same interval grid, their implemented angle-law KL is analytic.
If p_t,j is the teacher segment probability and m_t,j the teacher mean fraction
along that segment, then

    KL(t || s) = sum_j p_t,j * [(g_t0-g_s0) + (d_t-d_s)*m_t,j]
                - log Z_t + log Z_s.

This identity and its stationary matching gradient pass independent quadrature
checks. Averaging the conditional KL over the shared random-circle distribution
bounds the KL of the resulting marginal direction kernels by data processing,
including a shared failure mapping. This is standard information theory, not a
new theorem about molecular equilibrium or a bound for a coupled graph exchange.

Force and finite-work prediction remain separately measurable. A locally good
surrogate may still extrapolate poorly along a long feasible arc, so the next
gate is actual energy/Metropolis evaluation before training or superiority claims.

## Qualification and next steps

Seven targeted tests cover interval completeness, inverse CDF, sphere density,
parameter gradients, exact arc KL, known-target MH with an adverse uncorrected
control, and molecular graph/symmetry preservation. A geometry-only screen has
14,016 draws over all 438 real training contexts and four scoring choices. Every
draw preserves the original graph and has finite reverse density; independent
full-draw replay and first-draw density quadrature are recorded separately.
This validity repair is supplied by the geometric construction, not by learning.

The prospectively frozen true-oracle check is complete:824 first-draw endpoints
on206 FIT contexts,1,648 raw calls, full replay and independent raw-energy MH
checks. Equal composition means after parent means give fitted-score acceptance
0.7248 and expected potential change-0.0453 eV; uniform arcs give0.2478 and-0.1440
eV. Uniform arcs also have much greater accepted squared movement. These are
root-only TRAINING checks, with per-context scores held fixed under reversal.
High acceptance and local fit do not qualify a useful learned proposal. The
reproducible summary is `scripts/research/summarize_arc_oracle_feasibility.py`,
with `runs/arc_oracle_feasibility_summary_v2/results.json` reproducing the preserved
v1 summary exactly to1e-12.

## Complete joint integration

`cfm_mol/joint_arc_geometry.py` implements coupled graph, radius and two-root
regeneration. Both radii are drawn before directions. Each angular law uses its
actual partial context; unsupported passive arrangements fail once. Both angular
densities and Cartesian radial factors enter the coordinate density. The public
`joint_chemical_transition` checks the target's actual graph and uses the complete
forward/reverse density and eligible-action count ratio before MH.

The decoder samples one fair root order, retaining failures, but evaluates the
candidate density as the marginal0.5(q_order0+q_order1). Reverse density uses the
same marginalization. Both circle orientations are already included within each
root law. This standard mixture marginalization improves probability flow over
retaining order as an auxiliary variable; it is not a new general MH theorem.

Five targeted joint tests cover symmetry, marginal-order probability flow,
full six-coordinate chart volume, the public query ledger, and a complete known
joint target initialized by independent exact rejection sampling. The volume
check includes((N-2)/N)^(3/2)*exp(3ell1+3ell2)*|sin(theta1)sin(theta2)|. The fixed
COM factor cancels in the same-size reverse ratio. Root-only tests do not replace
these complete-joint checks.

The real training-start screen and independent audit are now complete:
`runs/joint_arc_support_v1` and `runs/joint_arc_audit_v1/results.json`. Each of
three arc scoring rules has745/768 supported positive-reverse proposals versus
legacy site's690/768. All3,072 attempts and random streams replay, including23
failed arc partial/empty attempts per method. Independent quadrature checks750
forward/reverse densities with maximum log error4.27e-14. Uniform arc order
marginalization rescues157 proposals with zero retained-order reverse density.
This remains a geometric support result, with no new neural weights or energy calls.

The next complete-chain TRAINING experiment is frozen in
`research/evidence/joint_arc_budget_protocol_v1.json`:48 FIT parents, four
compositions, two seeds, legacy site64 versus uniform/site arcs,128 raw queries
per parent and36,864 maximum additional calls. Common local and force-rotation
moves remain fixed. Producer46176878 and audit46176906 are submitted; refresh
Slurm before interpreting their status. The pilot does not establish equilibrium,
blind generalization, AI novelty or scientific submission readiness.
