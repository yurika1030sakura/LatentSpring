# Delayed acceptance: prior art and a possible query-cost diagnostic

The completed action/geometry training gives no positive joint internal point
estimate. Independent audits have completed and confirm these metrics. Do not change those
models or their frozen comparison. A separate possible investigation is reducing
expensive energy calls for proposals that a cheap model can screen, while
retaining an exact correction. No such new molecular method is implemented here.

## Primary literature checked September 12, 2026

- Banterle, Grazian, Lee and Robert, *Accelerating Metropolis-Hastings algorithms
  by Delayed Acceptance*, Foundations of Data Science 1 (2019), 103–128,
  https://www.aimsciences.org/article/doi/10.3934/fods.2019005 . Acceptance can be
  factorized into sequential tests; early rejection trades fewer target
  evaluations against reduced acceptance. Their work includes efficiency and
  optimization analysis. Generic factorization and query saving are prior art.
- Cao, O'Leary-Roseberry and Ghattas, *Derivative-informed neural operator
  acceleration of geometric MCMC for infinite-dimensional Bayesian inverse
  problems*, arXiv:2403.08220v2 (2024),
  https://arxiv.org/abs/2403.08220 . They combine a derivative-trained neural
  surrogate with delayed-acceptance geometric proposals and analyze costs.
  Neural screening plus derivative supervision is therefore not new by itself.
- Yu and Wang, *Accelerating Bayesian Phylogenetic Inference via Delayed
  Acceptance Sequential Monte Carlo with Random Forest Surrogates*,
  arXiv:2605.09506 (2026), https://arxiv.org/abs/2605.09506 . Their surrogate
  predicts move-induced likelihood changes using topology/branch features and
  screens tree moves inside SMC. Even learned screening of structural moves
  has close prior art. Abstract checked; this is not a full-paper audit.
- Bon et al., *Bayesian Score Calibration for Approximate Models*, JMLR 26
  (2025), https://www.jmlr.org/papers/volume26/24-1179/24-1179.pdf . This search
  result concerns transforming approximate posterior samples using scoring
  rules. It is not a direct implementation reference for a two-stage molecular
  MH screen; do not conflate its score with our physical force or MH ratio.

## Mathematical construction to test, not a new theorem

For a supported physical proposal with the complete log ratio R(x,y,a), any
finite cheap log factor s satisfying s(y,x,inverse(a))=-s(x,y,a) gives

    alpha_1 = exp(min(0,s)),
    alpha_2 = exp(min(0,R-s)).

The product is at most the original MH acceptance. Its forward/reverse ratio is
exp(R), so paired detailed balance is retained. Compute the expensive potential
only after the first test passes. An arbitrary classifier used to discard
proposals without this correction does not inherit that property. Fixed-step
stationarity remains distinct from the distribution at a query-budget stopping
rule. Source charges, spins, support and the paired inversion oracle stay fixed.

For the existing behavior-pair population, expected signed work is
E[-DeltaU alpha_1 alpha_2] and scored-proposal raw cost is E[2 alpha_1].
Unsupported attempts stay in both empirical denominators. A cheap model or its
training must not access the unseen proposed oracle energy before deciding
whether to query it. A factor built from the actual R is only an inadmissible
information diagnostic, never a measured algorithmic gain.

First quantify how much FIT-only oracle expenditure is associated with rejected
physical moves, and whether timing information can support a wall-time claim.
This can use cached outcomes without new physical calls or new fitting. Keep
internal selection, fresh follow-up and all other evaluated outcomes out of
fitting. Existing checkpoints trained on those parents cannot be silently reused
as held-out predictors. If a screening experiment is justified, freeze a cheap
physical surrogate and a simple learned baseline alongside any neural model,
then test actual chains and end-to-end costs. Fewer oracle calls alone does not
prove faster or more independent samples. Standard delayed acceptance alone
would not supply the missing ICLR novelty.

## FIT-only cost census completed

`runs/joint_query_cost_diagnostic_v1/results.json` retains all 1,242 FIT attempts,
including 56 unsupported attempts. The 1,186 scored joint proposals used 2,372
raw calls out of 9,216 calls for the corresponding complete physical trajectories.
Their parent/replica/composition-balanced expected acceptance conditional on
scoring is 5.35%. The unweighted expected calls attached to rejected joint moves
are 2,244.0, or 24.35% of recorded complete-trajectory calls. This is limited
headroom for this one move family; preparation and learning costs are additional.
It is not a measured achievable saving or a prediction for changed chains.

An explicitly inadmissible diagnostic uses the actual MH log ratio, clipped to
magnitude log(16), as the first factor. It preserves acceptance and suggests that
joint-query screening has information value. It uses the unavailable proposed
energy and cannot be deployed as a cheap method. Actual saved calls are zero.
Separate energy-RPC and geometry timings were not recorded, so a wall-time
speedup cannot be inferred by subtracting counts.

A bounded candidate, if pursued, should learn an antisymmetric paired-state
factor from only the 36 FIT parents, using source and already-proposed geometry
before the energy request. Exact reverse pairing must hold, including perceived
graph and electronic state. Start at s=0 so the original physical kernel is
recovered. Compare a cheap declared physical surrogate, a small fitted linear
screen and a neural paired-state screen; keep the base proposals, data and
factor bound fixed. Optimize the signed cost-adjusted accepted-work objective,
with no outcome-based acceptance-label threshold or uncorrected rejection.
Require real raw-call and full wall-time comparisons before production. A
successful screen would still need a distinct, useful molecular contribution
beyond established delayed acceptance to support an ICLR claim.

## Bounded implementation and frozen training

The screen and actual early-query dispatch are now implemented in
`cfm_mol/delayed_acceptance.py` and `joint_chemical_geometry.py`. The latter keeps
`valid` as geometry/reverse-support status and uses `scored` for actual oracle
queries when screening is enabled. First-stage rejects retain their coordinates
and failure denominator, but never acquire an oracle target ratio. A zero screen
exactly reproduces the old random stream, raw queries and resulting states.

The fixed physical screen combines squared covalent-radius-normalized bond
strain (10 eV coefficient), nonbonded inverse-distance power12 repulsion (1 eV),
and the exact declared COM restraint, plus the known complete proposal ratio.
These are heuristic surrogate energies. The linear screen fits four coefficients;
the neural screen adds a shared symmetric action-encoder readout difference
between the two orientations. Both learned models initialize at zero; neither
loads a prior checkpoint. The final odd tanh factor is bounded by log(16).

`research/evidence/delayed_screen_training_protocol_v1.json` freezes two seeds
per learned model, 300 steps, batch eight, the same FIT-only split and a signed
cost-adjusted two-stage objective. Zero/physical controls and every failed
attempt are retained. There are no new oracle calls. Independent audits evaluate
both factors with NumPy on every supported recorded pair and check detailed
balance, costs and sensitive trained-head gradients. Read NEXT for live jobs.

A separate fake-worker regression exposed TextIO buffering before select in the
tensor EnergyOracle. The existing NumPy oracle's byte-buffer reader fixes this;
original numeric outputs and requested/acknowledged counters remain. Energy RPC
wall time is now recorded separately, including failed calls, to support future
complete-cost comparisons. This does not retroactively supply timing for old
runs or qualify a sampling advantage.
