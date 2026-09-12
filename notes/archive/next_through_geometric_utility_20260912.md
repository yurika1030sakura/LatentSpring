# Next research actions — September 12, 2026

Read `research/ACCEPTED_UTILITY_STATE_20260912.json` and `research/STATUS.md`.
Use `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`; never write to home.
The ICLR objective remains active and scientifically unachieved.

1. The bounded geometric accepted-utility implementation, two-seed training and
   fresh-proposal audits are COMPLETE. The offline proxy improves about 2.7%;
   actual fresh proposals do not establish utility gain. Do not scale these
   frozen weights. Preserve the scalar-chain and stronger-physical-control
   failures. See `runs/utility_onpolicy_summary_v1/results.json`.
2. Implement `notes/bounded_action_geometry_candidate_v1.md`. Read the existing
   `cfm_mol/chemical_policy.py`, `notes/chemical_policy.md` and
   `notes/chemical_policy_decision_v2.md` before editing. Reuse its symmetric
   action features while preserving old APIs/checkpoints. Learn only the
   conditional legal-action law; keep the local/rotation/joint schedule fixed.
3. Use full forward and reverse action-plus-coordinate densities in the utility
   identity. With full densities, the separate action-log-ratio argument is zero:
   never count the forward/reverse eligible-action counts twice. Check inverse
   eligibility, normalized probabilities, atom/action symmetries, exact
   finite-state utility with unequal counts, and actual-map parameter gradients.
4. Freeze a small two-seed ablation protocol before optimization. Compare
   action-only, geometry-only and joint at equal total likelihood-ratio bound,
   physical initialization, data and optimizer budgets. The proposed bounds
   are action-only B_a=1; geometry-only B_g=0.5; joint B_a=0.5, B_g=0.25,
   each giving exp(2 B_a+4 B_g)=exp(2). This is not yet a frozen experiment.
5. Reuse only the protected physical behavior dataset and its fixed 36 FIT /
   12 internal-selection parent split. Retain all 1,671 attempts, including
   74 failures. No new physical queries are needed for initial implementation
   or offline fitting. The data ledger is 12,288 trajectory + 5,822 preparation
   = 18,110 raw calls. Keep signed accepted work, query cost and constitutional
   movement distinct; do not switch the primary metric after seeing outcomes.
6. Any offline signal still needs fresh measured sampling validation before
   molecular scale-up. Exclude fresh follow-up outcomes, prior evaluated
   six-composition coordinates, the 48 withheld-parent scalar trajectories,
   and all 722 reserved outcomes from fitting. Strong physical and representative
   learned-generator baselines, realistic reuse costs and independent final
   evaluation remain required on the declared scope.

All current BGFM jobs are terminal (46186496, 46186856, 46186968, 46189583,
46189707). Re-query Slurm; do not restart historical cancelled dependencies.
Latest paper build: `runs/verification/accepted_utility_20260912/main.pdf`;
build record: `research/evidence/accepted_utility_build_20260912.json`.
Prior NEXT: `notes/archive/next_through_scalar_chain_20260912.md`.
