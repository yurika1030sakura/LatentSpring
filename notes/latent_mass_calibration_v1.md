# Component-mass calibration with coupled reference generators

Status: implemented and tested on constructed targets with exact probabilities.
This is a bounded mechanism study, not a validated molecular generator or an
established distinct AI contribution. Current results and live job handles are
in `research/LATENT_MASS_STATE_20260913.json` and `research/NEXT.md`.

## The question and the learned object

Can we estimate the relative probabilities of disconnected components more
accurately by learning which latent samples to query together? A component is a
region within one fixed target, not a different chemical composition whose
arbitrary energy offset can be compared without a chemical-potential model.

For disjoint regions S_i, let q_i be a normalized, evaluable reference generator
whose support covers the target restricted to S_i. With reduced potential U,

    A_i(z) = exp[-U(F_i(z))] / q_i(F_i(z)),
    Z_i = E[A_i(z)],      p_i = Z_i / sum_j Z_j.

If support coverage fails, the integral is over the covered support, not the full
physical component. Overlapping reference generators need a separate partition
or an appropriate bridge/mixture estimator; one cannot simply add their Z_i.

We keep the reference generators fixed and learn a coupling C of their Gaussian
latents. Every marginal of C must remain standard Gaussian. The estimates of
each Z_i then retain their expectation, while their covariance changes. For two
components and finite second moments, the asymptotic variance coefficient of the
log-ratio estimate is

    Var(A_0)/Z_0^2 + Var(A_1)/Z_1^2 - 2 Cov(A_0,A_1)/(Z_0 Z_1).

The marginal terms are fixed. We fit scalar residual surrogates for log A_i from
queried values and maximize their predicted cross-moment. Normalization factors
used in fitting are estimates from the surrogates; exact Z_i is evaluation-only.
This changes the noise correlation of calibration, not either marginal sample
law or the marginal distribution of importance weights. Updating the component
mixture probabilities subsequently changes the mixture generator.

The generic ratio-estimation/coupling idea is already in
[Branchini and Elvira, GenSNIS](https://arxiv.org/abs/2406.19974).
Gaussian-preserving rearrangements of flows are also established; see
[Morel et al., Turning Normalizing Flows into Monge Maps with Geodesic Gaussian Preserving Flows](https://openreview.net/pdf?id=Fuyh90URIU).
Neither generic idea is claimed new here. A molecular implementation would need
a useful additional contribution and evidence beyond this toy construction.

## Exact finite-map implementation

For two Jacobi vectors z=(u,v) in R^(2x3), s=||u||^2+||v||^2, define

    T_theta(z) = R_phi(s) z,

where R rotates the two vector channels by the same angle in every spatial
coordinate. `ShapeTwist` predicts phi from log(s/6). Radius is invariant, so the
inverse negates the angle. In the derivative, the additional rank-one term has
determinant factor 1 because grad(phi) is radial and the rotation generator is
tangential. Thus |det DT|=1 and ||T(z)||=||z||, preserving the Gaussian density.
A fixed reflection of one channel also preserves it. This is an exact finite
map, without a trace estimator or ODE discretization correction.

The map commutes with a common O(3) transformation of both vectors. Arbitrary
permutations of identical atoms are not guaranteed to commute with the chosen
Jacobi basis; do not claim a general permutation-equivariant molecular layer.

Controls cover identity/common random numbers, independent Gaussian draws,
optimized constant rotations/reflections, and general O(3)-equivariant Gaussian
cross-covariance S tensor I_3. A simple eight-bin radial rotation/reflection also
preserves the Gaussian measure: each radius bin is invariant under its map.
Bin discontinuities have measure zero. This is a strong nonlinear control.

## Constructed reference and known truth

Three COM-centered points give six intrinsic degrees of freedom. For a chi_6
radius r, the reference map sends r to a_i + CDF_chi6(r), with a_0=1 and a_1=3.
It maps Gaussian latents to disjoint unit-width shells and has normalized
intrinsic density q_i(x)=1/(pi^3 ||x||^5). Interpolating its radial map with the
identity gives a monotone conditional flow-matching reference. The SciPy CDF
implementation is a numerical reference, not a differentiable training layer.

We set f=(||u||^2-||v||^2)/s and log A_0=f. The second log weight is
log(0.4)+2.5 f(T_psi^{-1} z). The two cases use a constant angle or
psi(s)=1.1 tanh(1.5 log(s/6)). These cases deliberately encode a twist; they are
not independent molecular discovery problems. Under a standard Gaussian, f has
a semicircle density on [-1,1], with M(k)=E exp(k f)=2 I_1(k)/k. Therefore

    Z = [1.130318207985, 0.805349198492],
    p = [0.583942367476, 0.416057632524].

The known twist gives a comonotone weight coupling and minimizes the asymptotic
log-ratio variance among fixed marginals. It is not a proven finite-sample
mass-MSE lower bound. None of these COM point clouds has a chemical-validity
certificate; they are not evidence of valid 3D molecule generation.

## Completed offline result and its limitations

The frozen protocol fits each case using 512 labels per component, then evaluates
independent calibration draws. Three fit runs and 1,000 calibration repeats per
budget are retained. At 64 pairs (128 evaluation calls), mean nonlinear-case
mass RMSE is 0.03884 independent, 0.03504 identity, 0.03494 optimized constant,
0.03532 Gaussian, 0.02285 neural, 0.02281 binned, and 0.02226 oracle.

Nonlinear coupling helps on this constructed nonlinear case, but the simple
binned method matches the neural model. The constant-angle case gives essentially
no neural increment over the optimized constant map. The primary experiment and
18,000 supplementary binned estimates replay. These comparisons exclude the
1,024 fitting labels per case/run from their inference budgets, so they do not
establish an end-to-end query saving. Repeated calibrations on one fixed target
are Monte Carlo evaluations, not different molecular conditions. Some latent
fitting banks are reused across components of adjacent fit seeds; they are not
independent molecular replications.

There is also an unfavorable distribution metric which must remain visible.
If q_i has conditional reverse KL K_i relative to the target in S_i, optimizing
only mixture probabilities by reverse KL gives w_i proportional to Z_i exp(-K_i).
For this example the variational optimum is [0.714286,0.285714], rather than p.
With the conditional shapes held fixed, joint reverse KL is 0.32398 at that
variational mixture and 0.36269 at the correct component masses. Better mass
calibration alone is not a better complete Boltzmann generator.

## Equal-total-budget online test

`online_latent_mass_protocol_v1.json` freezes 32 complete adaptive histories per
case, six methods, and four batches of 64 pairs. Every method spends 512 target
evaluations per history. Adaptive methods start at identity; a coupling fitted
from completed batches is used only on fresh Gaussian draws in the next batch.
All original target values remain in the normalizer sums. Old samples are never
remapped with a new coupling. With conditionally preserved Gaussian marginals,
the tower property gives unbiased normalizer sums in exact arithmetic. Ratios,
logs and component probabilities still have finite-sample bias. This is standard
adaptive-estimation reasoning, not a new theorem.

The audit checks every stored base draw, target weight and prefix estimator. It
also fully retrains and replays the first history of each chunk/case/method,
chosen independently of outcomes: 48 of 384 complete histories. Compute time is
reported separately from target-call cost. The primary readout is mass RMSE after
all 256 pairs; the simple nonlinear binned control remains mandatory.

## Completed online result and decision

All four experiment chunks and four audit chunks completed with exit0:0. The
summary is `research/evidence/online_latent_mass_summary_v1.json`, the flat table
is `research/evidence/online_latent_mass_table_v1.csv`, and the PDF with paired
uncertainty is `research/figures/latent_mass_v1/online_latent_mass_v2.pdf`.

At512 total target calls per history, including learning, mass RMSE is:

| Method | Constant case | Nonlinear case |
|---|---:|---:|
| Independent | 0.020724 | 0.016519 |
| Shared latent | 0.015757 | 0.016149 |
| Constant rotation | 0.012450 | 0.017523 |
| Gaussian coupling | 0.012459 | 0.017964 |
| Neural twist | 0.014276 | 0.012027 |
| Eight radial bins | 0.015247 | 0.014159 |

On the nonlinear case, neural RMSE is27.2% lower than independent and15.1% lower
than binned. However, paired neural-minus-control MSE intervals are
[-0.00027530,0.00000922] versus independent and[-0.00013219,0.00001322] versus
binned. Both span zero. Only the comparison with the fitted constant rotation
excludes zero among the five nonlinear-case controls. There are32 histories per
case; these are descriptive bootstrap intervals without multiplicity correction.
On the constant case, constant/Gaussian maps have lower point error than neural.
Do not promote the best selected comparison to a general superiority claim.

Full-run CPU times average approximately0.002s independent/shared-latent,
0.99--1.24s constant,1.71s Gaussian,1.63--1.64s neural and2.00--2.03s binned.
The toy oracle is cheap, so query reduction is not a wall-time acceleration here.
The primary experiment uses196,608 analytic target evaluations; audit weight
recomputations and48 complete-history replays are additional verification work,
not free production evidence. Of those48 histories,32 contain adaptive fitting
and16 are static controls. All196,608 saved weights and all prefix estimates
replay. No new molecular oracle calls were made.

Decision: retain this exact-marginal coupling as an estimator prototype. Its
nonlinear mechanism works on the designed case, but unique neural novelty and
molecular usefulness remain unestablished. It leaves marginal importance-weight
collapse unchanged and cannot by itself solve the original generator's sample
quality/coverage problem. Do not scale toy architectures or tune targets to force
a win. The main paper still needs a distinct useful learned-distribution advance
or a defensible contribution established on a controlled molecular task.

## Molecular transition boundary

The production FM64 endpoint plus noise has no established absolute q. The
corrected clamped q0.95 describes a different sampler and cannot be assigned to
those production samples. Refining an implicit source with EACF does not recover
the unknown source likelihood. Historical EACF support reports concern refiners,
not a qualified standalone normalized reference bank. Historical triatomic
references also differ in charge/spin conventions, potential symmetrization,
temperature or finite support. Reusing any of them requires explicit alignment.

Do not launch a broad molecular calibration sweep from those mismatched sources.
A defensible next molecular test needs one explicit normalized reference law,
one fixed composition/electronic sector and target, and a modest independent
probability reference or a precisely narrower estimator claim. It must compare
simple nonlinear and learned-generator controls and include preparation costs.
The present result alone does not justify a large molecular implementation or
another sweep of toy architectures to force a neural win.
