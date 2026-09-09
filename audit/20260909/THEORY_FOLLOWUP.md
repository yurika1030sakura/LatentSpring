# Follow-up theory and manuscript audit

The September 8 archive stays frozen. These corrections apply to the active
research manuscript. A top-level caveat does not excuse contradictory claims
inside a proof, assumption, capability table or reproducibility paragraph.

## Corrected statements

1. **An unbiased stochastic trace is not an unbiased squared residual.**
   The trace-noise covariance contributes a parameter-dependent quadratic
   penalty. Common probes replace a Jacobian-magnitude penalty by within-parent
   Jacobian dispersion. Gaussian probes use the symmetric part of the Jacobian;
   `2 ||J||_F^2` is generally wrong for nonsymmetric J. Exact moment quadrature
   tests verify the Gaussian and Rademacher formulas.
2. **A continuous CNF density and the finite sampler's exact density differ.**
   State integration plus divergence quadrature need not equal the discrete
   map's exact log determinant. Normalization applies to the ideal global
   diffeomorphic flow under its assumptions, not automatically to the computed
   scalar. The archived wrong head adapter and absent COM projection cannot be
   fixed by increasing step count.
3. **Read-out sensitivity does not imply gradient-noise divergence.**
   The score map multiplies velocity error by t/((1-t) sigma^2). Its covariance
   is multiplied by the square of that coefficient, conditional on the input.
   Velocity error may itself vanish with time; parameter gradients also depend
   on the network Jacobian and residual. The old claimed universal bias-variance
   lower bound used big-O expressions as lower bounds and has been removed.
4. **Endpoint force regression is a conditional expectation.**
   Its squared-loss target is E[F(X1)/kT | Xt]. Equality to the realized force
   requires force measurability, not recovery of every coordinate of X1. With
   equilibrium data and vanishing integration-by-parts boundary terms, it is
   `t * score(p_t)`, not generally `score(p_t)`. Distribution mismatch alone
   cannot prove different targets at every t: for Gaussian data variance a
   and target variance b, the targets coincide at a probe time when b=t*a.
5. **Fixed COM does not guarantee a finite partition integral.**
   Dissociating fragments remain unconfined. The target domain, confinement,
   symmetry measure and electronic state must be specified before global Gibbs
   claims. Only coordinate-independent factors cancel in a parent variance;
   a geometry-dependent rotational/internal-coordinate Jacobian does not.
6. **The old surrogate is not an established likelihood gradient.**
   Its notation now explicitly uses the raw-head scalar and frozen-trajectory
   update. A recomputed stop-gradient update is not guaranteed to descend the
   underlying scalar. The new discrete adjoint differentiates the numerical
   objective and is documented separately.
7. **The capability table cannot substitute for baselines.**
   The old matrix incorrectly assigned a validated CNF likelihood to the joint
   pipeline and used that to claim no direct comparator. It has been replaced
   with an auditable table of the project's distinct sampler and density objects.
   Models without tractable likelihoods still admit independent sample-quality,
   energy, coverage and compute comparisons.

## Evidence and limits

The edited files are `paper/sections/A1_proofs.tex`, `A2_details.tex` and the
reproducibility paragraph in `paper/main.tex`. The main text remains nine pages;
compilation checks citations and references. Mathematical revisions do not
upgrade the old scalar experiments into physical or likelihood evidence.
The new independent sampling and numerical results are in Appendix D and
`research/evidence/`, with failed integrations and evaluations retained.
