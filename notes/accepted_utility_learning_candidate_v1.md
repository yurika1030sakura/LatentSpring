# Candidate after the scalar predictor fails the molecular comparison

The scalar work and work-plus-force models have no demonstrated complete-chain
energy advantage at equal inference cost and lose the data-cost comparison.
Do not scale or repeat these frozen models. Canonical move diagnostics also show
no clear gain in expected accepted constitutional changes. The finite-candidate
confidence diagnostic is not a continuous conditional-KL estimate and does not
prove the cause of this failure.

A bounded next investigation is direct learning of actual Metropolis-corrected
proposal utility, using existing FIT-parent physical proposal pairs. This is a
candidate to implement and test, not a successful replacement. Accepted-movement
objectives, importance sampling and bounded residual scores are established ideas;
see the existing Timewarp/L2HMC review and `nonlocal_arc_prior_art_20260912.md`.

## Exact conditional identity to verify

Let x be a fixed training source state, a an eligible uniformly sampled action,
y a behavioral proposal, q_b the recorded behavioral coordinate density and
q_theta,f / q_theta,r the learned forward/reverse coordinate densities. With
A = -(U(y)-U(x))/kT + log(N_forward/N_reverse), an importance-weighted accepted
utility r(x,y) is

    r(x,y) * min(exp(log q_theta,f - log q_b),
                 exp(A + log q_theta,r - log q_b)).

This follows by multiplying the importance ratio by the actual MH acceptance.
It is a standard probability-flow identity, not a new MH theorem. Its expectation
is the learned kernel's one-step utility for the chosen FIXED source-state
population, provided behavioral support covers the learned successful proposals.
It is not a proof of better long trajectories or of equilibrium mixing.

Both coordinate densities must include all actual partial contexts, both decoder
orders, circle preimages, radial factors and numerical support rules. Learning
only an energy difference or omitting the reverse density would not implement
this objective. Start with differentiable OBSERVED-density evaluation; sampling
should retain its existing no-gradient boundary. Verify derivatives against an
exact finite-state example and finite differences of the actual molecular map.

Use all recorded attempts in denominators. Unsupported or zero-reverse-density
attempts contribute zero utility when the same support rule makes their learned
MH flow zero. Do not condition training on accepted or merely scored proposals.
Audit any numerical-failure support mismatch before using an old behavioral run.

## Guard against offline overfitting

Start from the exact physical site-arc proposal, with no old learned checkpoint.
Consider a bounded residual LOG SCORE, not a claimed physical-energy predictor.
If its pointwise residual magnitude is bounded by B, endpoint-linear interpolation
preserves that bound. Each normalized root law then changes density by at most
exp(2B), and a two-root order-marginalized joint proposal by at most exp(4B), with
the same geometric/radial support. This elementary likelihood-ratio bound is not
a target-error or efficiency guarantee. Verify it on the implemented law.

Inspect actual importance weights, effective sample size and per-parent utility;
a large empirical objective driven by a handful of behavioral edges is a failure.
Do not silently clip weights and call the resulting objective unbiased. Any
truncation, trust penalty or residual bound must be fixed and reported explicitly.

## Bounded next implementation

1. Audit the already completed FIT-parent physical site-arc trajectories from
   `runs/joint_arc_budget_v1`: four compositions,48 FIT parents,two seeds. Freeze
   source hashes and a new energy-blind internal parent split before fitting.
   The previously evaluated48 withheld-parent trajectories cannot become fitting
   data for this candidate. Reserved722 outcomes remain untouched.
2. Implement differentiable observed joint densities and the utility identity.
   Use an exact finite-state test, actual-map finite differences, support checks,
   and an importance-weight audit before any optimizer is run.
3. Freeze a small two-seed offline comparison from the physical initialization.
   Report signed accepted potential change and constitutional-move utility as
   distinct quantities; do not switch the main metric after outcomes. No new
   physical queries are required for this exploratory fitting stage.
4. Charge the behavioral trajectory construction when the data become training
   inputs. Site-arc FIT trajectories cost12,288 raw calls, in addition to their
   relevant preparation; they cannot remain free merely because they were first
   computed as a physical diagnostic. Do not inherit the old14,862-call overhead
   unchanged if a new training-data recipe is used.
5. Only after internal numerical and statistical qualification freeze a new
   learned-chain comparison against physical site arcs and appropriate additional
   learned/physical controls. The present failed scalar result remains unchanged.
