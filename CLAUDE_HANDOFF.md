# Claude continuation — finite chemical work and edit selection

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `research/NEXT.md` and current state before action.
Never write home or merge the FlowMol/oracle environments. User priority is AI
framework plus rapid evidence; optimizer recovery remains paused.

Current source of truth: `research/CHEMICAL_WORK_POLICY_STATE_20260913.json`.
The new paired finite-work model and normalized forward/reverse edit policy are
IMPLEMENTED, trained and fully audited. Both seeds and all linear/graph/3D
controls are retained. On18 existing evaluation-only parents, exact one-step
corrected utility is0.070621 eV for uniform selection,0.062508 for cached force,
0.121892 for linear bond learning (seed mean), and0.088391 for3D learning.
Linear-minus-uniform is+0.051271 eV, descriptive parent interval[0.011624,0.099495].
The3D-minus-linear interval spans zero and its point is worse by0.033501 eV.
No distinctive useful neural contribution is established. These are one-step
expectations over actual molecular catalogues, NOT complete-chain, equilibrium,
wall-time or data-cost superiority. The evaluation uses498 raw queries,231 valid
edits and20 failed edits;1,848 physical MH ratios and the complete trace replay.
No new training labels were queried. The training audit replays all12 models
and5,364 pair reversals. Jobs46258095/46258563/46258596 are all COMPLETE0:0.
Read current NEXT; retain the simple learned policy as a strong baseline. Stop
scaling the frozen3D recipe. AI novelty remains the priority, optimizer recovery
stays paused, and evaluated parents/722 reserved outcomes remain outside fitting.
The full ICLR goal remains active and scientifically unachieved.

The next action is a bounded representation experiment that includes the
successful additive work baseline and learns only the non-additive3D part.
Do not call a generic neural residual or standard informed MCMC new by itself.
All code, exact model paths, split restrictions and validation commands are in
NEXT and `notes/chemical_work_policy_candidate_v1.md`. No fresh data is required
to start. Keep every negative result. Do not fit the18 or original12 evaluation
parents or query the722 reserved outcomes. Full-chain gains and relevant learned
generator comparisons remain future evidence gates for the actual paper claims.
Authors/submission stay with user. No ICLR acceptance/readiness claim is warranted.
