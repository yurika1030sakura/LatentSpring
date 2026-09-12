# Current reconstruction: valid molecular states and reversible topology/geometry moves

2026-09-11. The ICLR objective remains active and unachieved. The original
48-arm experiment is finished and retained. The research direction is now a
support-aware molecular Markov sampler initialized by a flow-matching generator.
The current paper remains the audited development draft for the previous
exact-entropy refiner; it is not a submission-ready description of this new route.

September12 update: a separate current-candidate draft is now
`paper/angular_working.tex`; the older `main.tex` study is preserved. Read
`research/MASKED_ANGULAR_STATE_20260912.json` first. The masked angular guide and
its exact directional envelope are implemented, trained and evaluated. Tensor
angular acceptance reaches75/256 and58/256, but both replicas still fail the
difficult graph barrier and no total-cost learned advantage is established.
The next concrete experiment is `notes/joint_graph_geometry_design.md`.

Latest checkpoint: `research/INTERNAL_COORDINATE_STATE_20260911.json`. The symmetric
nonequilibrium paths, graph guidance and frozen-selector composition have now
been implemented and tested; they do not establish an overall advantage. The
force-informed S2 angular control produces the first difficult-parent transition
in one of two replicas, while the other still fails. Read
`notes/spherical_chemical_candidate.md` and the next bounded AI design in
`notes/masked_angular_learning_design.md`. The older prospective descriptions
below are retained as history where superseded.

## Why the original route is insufficient

The completed original campaign has48 qualified arms and866304 raw refinement
queries. All8 geometry conditions complete1792 attempts:1666 converged and126
failed. Small internal ablation effects do not overcome the adverse full EACF
and MALA/HMC comparisons. All failures and previous prefixes remain unchanged.

The composition-only energy target also admits fragments and different
constitutional isomers. The unrestricted temperature-exchange diagnostic found
three-component FM states with lower energy than the connected QC structure.
Its restrained-energy rank Rhat is2.62; it is not an equilibrium reference.
The connected/no-overlap diagnostic keeps configurations connected, but its
restrained-energy Rhat is1.78 and its FM/QC initialization families remain apart.

Chemical graph perception identifies a concrete barrier: generated warm starts
are predominantly FC[SH](F)(F)(F)F, while the QC-origin starts are
CS(F)(F)(F)(F)F. This is a change of constitutional identity, not just a torsion.
Do not interpret independently inferred octahedral stereo tags as different
connectivity identities. The v2 audit constructs conformers directly, avoiding
the XYZ scientific-notation parse failure in the preserved v1 audit.

In condition0,66/4096 generated training parents and8/512 development parents
pass the geometric domain. Only34/66 and4/8, respectively, also pass the present
RDKit graph assignment/sanitization. None of those four development molecules
has the QC reference connectivity. Geometric validity alone is not chemistry.
Source rejection and generation costs must remain in every denominator.

## Implemented continuous proposal

`cfm_mol/conditional_molecular_proposal.py` conditions all atomic updates on the
complete current state. Gaussian noise on the labelled unweighted COM subspace
passes through a learned low-rank collective map and heterogeneous nonlinear
centered point maps. Unlike the old static whole-element update, the current
coordinates remain available as context even for homogeneous elements.

For fixed context x, let A contain normalized learned collective vectors and
C be diagonal with sum|C_kk|<1. The linear noise map is L=I+A C A^T, with
intrinsic determinant det(I+C A^T A). Its collective vectors have zero total
translation and torque; the full proposal density still uses dimension3(N-1),
not a rotation-reduced dimension. Each point map has derivative B_i bounded
between .75 lambda I and1.25 lambda I. Centering its outputs contributes
det(mean_i B_i^{-1}) in addition to product_i det(B_i).

The resulting proposal has an explicit q_theta(y|x). Its numerical inverse
is differentiable, including context dependence in q_theta(x|y_theta).
Checks cover complete intrinsic Jacobians, inverse likelihood gradients,
reflection/permutation symmetry, batch independence,200 atoms, known-target
stationarity, and rigid-motion invariance of the movement observable.

These ingredients and determinant identities are established mathematics.
Architectural usefulness and novelty remain empirical/prior-art questions.

## Hard support changes both the target and the training contract

The geometric prototype uses a declared connected/no-overlap domain. A later
chemical variant also requires deterministic graph assignment at fixed charge.
Neither criterion is a quantum certificate; source spin labels stay separate.
All earlier unrestricted-target results keep their original meaning.

A full-support source followed by a diffeomorphism of all H has positive mass
outside a proper hard validity domain. Its KL to a zero-density target there is
infinite. Therefore the old finite relative-KL objective cannot simply acquire
an indicator penalty and retain its claim. Conditional Metropolis proposals
instead use explicit forward/reverse probabilities and reject unsupported states.
They need no source-density estimate. This also allows the FM initializer to be
improved separately without the old source-entropy-cancellation restriction.

`cfm_mol/metropolis_utility.py` implements the score gradient of accepted utility.
At fixed sampled x,y, with MH ratio r and acceptance a=min(1,r), the gradient is
E[a J (1_{r>=1} grad log q_forward +1_{r<1} grad log q_reverse)]. A baseline
depending only on the old state or past batches multiplies the forward score
for ALL proposals, including unsupported ones. The source of this identity is
ordinary score-function calculus; it is not claimed as new physics or a new
general theorem. Positive-mass acceptance ties retain the differentiability caveat.

Tests exhibit the missing boundary derivative in naive pathwise differentiation
through a hard indicator and verify the score formula against exact Gaussian
acceptance derivatives. Proposal samples must be detached before evaluating
these inverse-density scores. A reparameterized forward log density is not the
same score. This helper is implemented but has not trained a molecular actor.

## Learning evidence so far

The original expected-shape-jump objective improved accepted movement but not
mode mixing in the controlled Gaussian-mixture experiment. An oracle mode
coordinate improved diagnostic mode ESS but is not a deployable method.
The data-derived variant learns a slow direction in a fixed radial RBF family;
it uses no mixture identities during training. Its held-out correlation with
the true mode coordinate is .95--.98. This is not a general molecular encoder
or a global spectral-gap bound.

The longer frozen test includes inherited training in the156704-target-call
budget. A fixed .5 local-MALA/neural mixture gives nonlinear mode ESS
264.5/842.7 versus MALA250.7/107.3; the affine mixture gives274.5/209.6.
Seed variation is material and rank Rhat still exceeds1.01. These are controlled
toy diagnostics, not established molecular or universal sampling superiority.
Keep global-only failures and all affine/MALA/RWM controls.

## Implemented reversible chemical move and decisive diagnostic

`cfm_mol/chemical_moves.py` exchanges unlike monovalent terminal atoms between
their attachment sites. Anchor-relative vectors are rescaled by covalent-radius
ratios. If leaves i,j attach to passive anchors k,l, the positive scale factors
are s_i=(r_i+r_l)/(r_j+r_l), s_j=(r_j+r_k)/(r_i+r_k).
The inverse exchanges the anchor roles. Before COM centering the map is
translation equivariant, and its intrinsic absolute determinant is(s_i*s_j)^3.
The MH ratio includes that determinant and the actual reverse-action probability.
Tests check inversion, the full intrinsic Jacobian and atom/spatial symmetries.
This is a Monte Carlo move, not a physical reaction trajectory.

The first uniform-policy diagnostic uses local/exchange/local palindromic
cycles versus three local MALA moves, with identical warm starts. Each of the
four arms completes1544 new raw queries. Both exchange replicas convert both
FM-origin starts to CS(F)(F)(F)(F)F; neither MALA replica does. QC-origin
controls remain in that connectivity. Starting-reference computation is
additional and must not be hidden. This establishes a useful physical proposal
for the observed barrier, not AI novelty or complete equilibrium sampling.

## Next work

### September 11 learned-policy implementation checkpoint

The action policy is now implemented and trained, rather than only proposed.
`cfm_mol/chemical_policy.py` has 7106 parameters, invariant whole-state messages,
defensive local/exchange probabilities and explicit reverse family/action
correction. `cfm_mol/chemical_sampler.py` records every proposal, invalid reason,
paired raw energy/force result and acceptance random variable. See
`notes/chemical_policy.md` for the target and finite-table training objective.

The generated-training-only table job46038056 completed with5564 training raw
queries and404 development warm-up queries. All34/4096 eligible training and
4/512 eligible development parents are included. No QC/evaluation coordinates
trained the policy. Saved query arithmetic, all3078 proposals and full random
streams replay in `evidence/chemical_policy_table_rng_audit_v1.json`.

Both offline training replicas in job46038463 completed without additional
physical queries. Their empirical accepted utility rose from0.5222 to1.2441 and
1.2438; this is a training statistic, not a molecular sampling advantage. Both
models saturate the0.1 local-family floor. The evaluation therefore includes
fixed0.5-local and fixed0.1-local uniform-action controls.

Job46039029 (`runs/chemical_policy_eval_v2`) completed all six evaluation arms.
The v2 evaluation protocol was frozen before evaluation outcomes and corrects
the source preparation ledger: learning inherits4608 source queries, uniform
sampling only512. Training preparation adds5564 for learning; shared development
warm-up adds404. Uniform controls may run1482 steps versus256 learned steps.
Use actual total-cost prefixes, not nominal equal-step or maximum-budget labels.
Source neural generation and offline wall time remain additional distinct costs.

Both learned replicas reach reference connectivity at step1 for three of four
development parents. The remaining parent stays trapped in every arm. Against
the0.1-local uniform control, equal-step mean energies are mixed; all cheaper
uniform final endpoints have lower mean energy than the learned endpoints.
All45644 new evaluation queries, graph/proposal decisions and random streams
replay. No overall learned sampling/cost advantage is established; do not scale
this selector recipe on condition0.

The additional classical multiscale control46040976 is complete with1848/1838
queries. For the difficult parent, local acceptance improves to66/138 and62/132,
but neither replica crosses its connectivity barrier. This motivates testing
exchange coupled to geometry adaptation. `notes/nonequilibrium_chemical_candidate.md`
specifies an audited-path construction to implement next; it is not yet a result.

The all-eight-condition chemical census finds terminal-exchange coverage only
in conditions0/4. RDKit algorithm errors on metals and charged-mode rejection of
neutral radicals are retained separately. The experimental v2 radical fallback
passes basic methyl/ammonium checks, but25/32 preview assignments contain3--7
radical flags and are NOT quantum/spin qualified. This new support is not adopted
for production and does not relabel old trajectories.

Read `notes/chemical_policy_decision_v2.md` and
`research/CHEMICAL_POLICY_STATE_20260911.json` for the complete decision and next work.

1. Implement/test the symmetric nonequilibrium exchange/relaxation path and its
   complete reverse probability. Run a bounded physical control on the measured
   geometry barrier before training another network.
2. Learn only where the stronger physical controls leave a measured inefficiency.
   Validation/QC reference coordinates cannot train the generator or policy.
   Count every intermediate query, training and source-initialization cost.
3. Extend chemistry moves only where a measured barrier requires it; terminal
   exchange is a limited primitive. More general reversible fragment exchanges
   must carry their exact inverse, atom inventory, validity and COM Jacobian.
4. Establish sampling and chemical-validity performance on independent molecules
   and larger systems. The current connected reference is not yet qualified.
   Reserved722 outcomes remain untouched until a fixed method is ready.

Relevant prior art already includes [Timewarp](https://arxiv.org/abs/2302.01170),
[A-NICE-MC](https://arxiv.org/abs/1706.07561), [L2HMC](https://arxiv.org/abs/1711.09268),
[Neural Mode Jump Monte Carlo](https://arxiv.org/abs/1912.05216) and
[Markovian Flow Matching](https://arxiv.org/abs/2405.14392). Learned proposals, accepted-jump
objectives, mode jumps, Metropolis correction and ordinary change of variables
cannot be claimed as new ingredients by themselves.
