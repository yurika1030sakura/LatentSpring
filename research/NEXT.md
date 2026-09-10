# Next research decision

Goal active and unachieved. Previous goal turn: progress (implemented/calibrated
new features, ran exact-entropy molecular refinement and fresh confirmation,
and implemented the nonlinear constrained-flow primitive).263 tests pass.
All current BGFM jobs are complete. No broad calibrated sampler or validated
ICLR novelty is established.

1. Implement the invariant context conditioner and species coupling wrappers
   described in notes/species_coupling_adapter_candidate.md. Prefer the new
   pointwise centered convex primitive over a shared linear A-only internal map,
   which preserves too many within-type affine invariants. The primitive exists
   in cfm_mol/centered_convex_flow.py and has full Jacobian/inverse/gradient tests.
   Build and test internal-group and centroid-pair layers with complete context
   dependencies, COM, O(3), joint coordinate/element permutation, identity and
   inverse composition. Singleton and homogeneous limitations must be explicit.
2. This route keeps the FlowMol FM base and adds an invertible adapter with
   exact entropy change. Train only after real-interface tests, using physical
   energies/forces and exact log volume; do not use unqualified proposal scores.
   Then compare nonlinear/typed-linear/scalar at matched budgets and fresh
   evaluation. Retain all failures. No nonlinear molecular job has been launched.
3. Exact linear controls now provide a small confirmed relative-KL signal:
   typed -.10193+/-.02719 nat, scalar -.06356+/-.01809 on1024 fresh samples.
   Typed-minus-scalar -.03838+/-.01738. One trained seed/one condition only.
   ESS remains1.8435(base),2.2454(typed),1.7519(scalar) of1024. This is not target
   calibration, broad sampling improvement or established novelty.
   Full evidence: linear_entropy_adapter_comparison_v1.json and
   linear_entropy_adapter_confirmation_v1.json. New oracle cost including smoke,
   both training controls and confirmation is10560; inherited costs are additional.
4. Learned-feature score calibration still fails14-moment qualification despite
   beating random and typed-feature DSM baselines. Do not repeat the failed
   score-actor recipes. Keep neural_head_calibration_v1.json and every earlier
   score/CNF/work/covariance/SMC failure. Its16384 panel is development data.
5. Primary prior art includes EACF2023, Equivariant Finite Flows, Convex Potential
   Flows and Residual Flows, plus classic Boltzmann generators. The determinant,
   contraction and KL identities are not novel. Method novelty requires a useful
   distinct architecture/learning intervention and replicated total-compute gains.
6. Reserved722 conditions remain untouched. Eight new development conditions
   have a baseline but no candidate sampling result. AgBr2 references remain
   three-atom evidence only. Broad conditions/seeds and strong matched baselines
   are still required, as is a complete new-method manuscript.

Completed latest jobs: neural-feature smoke45849201/full45849360; linear smoke
45850658, typed45850990, scalar45850994, xTB45851844, confirmation45855978.
Never write home, alter shared FlowMol or merge environments. Keep original
charge/spin, bond-free OMol25, max_atoms200 and immutable source snapshots.
Do not mark the full ICLR goal complete or blocked from these partial results.
