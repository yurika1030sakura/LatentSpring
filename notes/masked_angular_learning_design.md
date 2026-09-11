# Next AI candidate: masked conditional angular guide

Status: design for the next implementation, not a trained model or a novelty
claim. The force-informed S2 control has the first difficult-parent transition
in one of two replicas (step163); the other remains trapped. Uniform angular,
nonequilibrium, fixed graph-guided and frozen-selector-composed controls did not
resolve it. Do not expand those failed recipes or call one physical success a
stable learned advantage.

## Concrete representation to implement

For a selected terminal atom i and anchor k, the update retains every passive
relative coordinate w_j=x_j-x_k, the radius r=||x_i-x_k||, atom identities,
electronic condition and perceived graph G. The input to the angular model must
REMOVE the current direction of i before any message passing, pooling or
geometric feature calculation. Replace the leaf coordinate by the anchor in
the context. Use anchor-relative passive coordinates and an explicit leaf/anchor
role marking; do not encode absolute atom indices as features.

The retained context c is identical at x and any valid same-graph angular
candidate y. Test this equality under the actual COM recentering operation,
joint atom permutations, translations, rotations and reflections. This property
is more important than network depth: leaking the old leaf direction invalidates
the normalizer cancellation below.

Use a small shared graph network to produce a polar vector eta(c) and a symmetric
traceless tensor A(c), giving an angular log score

    s_theta(u;c) = eta(c) dot u + u^T A(c) u.

The tensor allows antipodal/planar angular structure that a single equivariant
mean vector cannot represent in symmetric contexts. Construct vector and tensor
outputs from invariant scalar weights on passive vectors and their dyadic
products, preserving O(3) and atom-permutation covariance. Remove the isotropic
trace of A, which is an unidentifiable constant on S2. Bound coefficient norms
for stable rejection sampling and retain an isotropic initialization. Neither
Fisher-Bingham models nor equivariant tensors are new ingredients by themselves.

## Exact kernel without a support-normalizer estimate

Let D(c) be the same-graph valid angular set under the unchanged target validator.
Sample uniform directions and accept a proposal draw with probability

    1_D(c)(u) exp[s_theta(u;c)-M(c)],

where M is a proven upper bound (for example ||eta||+||A||_F). Stop at the first
success or after a fixed cap K. On exhaustion, retain the old state. Save every
trial and all random variables; count neural/geometry runtime separately from
physical oracle queries.

Writing Z(c) for the unknown one-trial acceptance probability, the generated
non-self transition density is proportional to

    exp[s_theta(u;c)-M(c)] * sum_{j=0}^{K-1} (1-Z(c))^j.

Both c and D(c) are identical in the reverse move. Therefore the unknown factor
and the bound M cancel in the angular MH ratio. The physical acceptance is

    -(U(y)-U(x))/kT + s_theta(u_old;c)-s_theta(u_new;c)
    + log p(reverse leaf|y)-log p(forward leaf|x).

Start with uniform leaf selection. The radial r^2 measure cancels as in the
existing S2 control. This supplies an exact RATIO for the kernel, not a normalized
conditional likelihood or full Cartesian endpoint density. Do not insert this
score into a KL estimator or claim an estimated molecular partition function.
Context invariance and this cancellation are standard conditional-MH reasoning;
the research contribution must be the architecture's demonstrated usefulness.

## Training using already qualified training data

Use only the generated TRAINING table/trajectories, whose raw paired forces and
electronic conditions have been audited. Never use development/QC coordinates
to fit the network. The table already stores2782 supported scored states from
34 original training parents, including local proposals and exchange candidates.
Keep parent/time correlation and source costs explicit.

A first bounded fit can use supervised angular force regression. The target
surface score is r(I-u u^T)F_COM,i/kT; the model surface score is

    (I-u u^T)(eta+2 A u).

At fixed masked context this is an analytic derivative. No oracle Hessian, source
density or conditional normalization constant is needed. This is pointwise force
regression on finite-time training data, not an equilibrium score-matching theorem
or proof of a correct learned density. Reject unsupported/mismatched records and
retain the actual inherited preparation cost. Freeze hyperparameters and parent
roles before inspecting development performance.

## Required tests and decision

1. Finite-difference/autograd surface-gradient checks, tensor/vector covariance,
   permutation and batch independence, and exact masked-context invariance.
2. A finite conditional-state model verifying detailed balance for the capped
   sampler with unknown Z. Include a negative control showing that leaking the
   old direction makes the omitted normalizer ratio bias the target.
3. Known S2 targets, support rejection/exhaustion accounting and complete
   proposal/acceptance/RNG replay before molecular use.
4. Compare frozen learned proposals with uniform angular, force-vMF and the same
   masked model without learned coefficients. Keep the existing chemical exchange
   and multiscale local stages fixed. Include training, source preparation,
   candidate-search time and every physical query.
5. Establish repeated molecular gains and independent conditions before writing
   this as the main paper. One composition, a positive toy, high acceptance or
   an exact kernel alone does not establish ICLR-level novelty or readiness.
