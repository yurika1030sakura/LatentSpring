# Prespecified neural-context affine control

The first two nonlinear-versus-typed comparisons confound a19,365-parameter
neural context with active point nonlinearity. A13-parameter global linear
control alone cannot identify which component causes the small advantage.

The implemented AffineSpeciesCouplingAdapter inherits exactly the conditioner,
parameter names/shapes, group/context decomposition, schedule and initialization.
It replaces each active point map with F(0)+DF(0)u. DF(0) is SPD under the same
coefficient bounds as the convex map. For a group internal update, centering
cancels F(0) and the log volume is(n_g-1)logdet DF(0); a centroid-pair update
uses logdet DF(0), retaining F(0) plus the same bounded learned shift. Inverse
is a3-by-3 linear solve. Context dependencies remain differentiable. The full
conditioned composition is still nonlinear and must not be described as a
global linear map. Some coefficient effects are coupled differently after
linearization; this is a mechanistic ablation, not proof of equal expressivity.

Fourteen full-model tests now apply to the convex and affine families, covering
complete intrinsic Jacobians, inverse, symmetries, batch independence, identity,
parameter gradients and200-atom reconstruction. Family-specific checkpoint kinds
prevent a saved affine map from silently loading as a convex map.

Before training, run the same2-update16-row real-oracle smoke(64 queries), with
init9161/selection9141. Require identity energy replay, finite gradients, trained
inverse and two complete intrinsic Jacobians. After passing, run exactly1000
updates, batch16, cosine .001 to .00001, same4096 training parents and256
development rows as the existing nonlinear candidate;16512 queries. Use both
previously prescribed pairs(init9161/selection9141 and init9162/selection9142).
No best-checkpoint selection or hyperparameter adaptation. Evaluate frozen maps
on the existing2048 confirmation rows; these outcomes are shared development
evidence for the mechanism, not independent final validation. Cost per trained
affine arm:16512 training/development+2048 confirmation oracle queries.

If affine-context control matches the convex candidate, do not claim that active
point nonlinearity caused the earlier gain. Retain it as a simpler competitor
and investigate the shared molecular decomposition on the full development panel.
If the candidate wins, assess uncertainty and broader replication before any
mechanism claim. Either result still requires strong EACF/sampling baselines and
does not repair the existing ESS~1/2048 failure by itself.

First training pair completed: affine development DeltaKL=-.21483+/-.06354;
convex-minus-affine=-.02008+/-.00628. On the stored2048-row confirmation,
convex-minus-affine=-.01332+/-.00252 nat, with2048 additional physical queries
and zero repeated base/convex energy calls. This supports a small active-map
benefit for the first training pair; the prespecified replica is still running.
Do not call this a replicated mechanism or a sampling-calibration result.
