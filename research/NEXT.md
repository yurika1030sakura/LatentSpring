# Current AI-method work — chemical edit choice

Read `research/CHEMICAL_WORK_POLICY_STATE_20260913.json` and
`notes/chemical_work_policy_candidate_v1.md`. User priority: differentiated AI
framework plus rapid molecular evidence, not perfect optimization of every case.
The full ICLR objective remains active and scientifically unachieved.

1. CLOSE the current learned-mobility superiority claim. On18 new parents,
   collective, bare edit and small root noise give mean128-query changes
   -0.60935,-0.59773,-0.61205 eV. Learned-minus-root-noise is+0.00270 with an
   interval crossing zero. Simple controls explain the earlier arc advantage.
   Preserve all positive/negative comparisons, including the original proposal
   signal. Do not scale or tune these mobility weights to manufacture a win.
2. Focus the next AI hypothesis on WHICH edit to query. In bare transfer chains,
   576/614 proposals have scored valid geometry,531/576 increase energy and55
   accept. This is an edit-choice opportunity, not proof of AI novelty.
3. The complete FIT work catalogue is DONE and audited:36 sources,492 eligible
   edits,447 valid candidates,45 failures,966 raw calls.26/36 sources have a
   downhill candidate. Valid candidates are69.1% uphill; source-force linear
   work has parent-balanced MAE3.185 eV. These are training diagnostics only.
   Paths: `runs/chemical_work_catalogue_plan_v1`,
   `runs/chemical_work_catalogue_v1`, `runs/chemical_work_catalogue_audit_v1`.
4. Implement the compact masked passive-context model: encode unchanged atoms,
   query both active placements through a shared conditional-energy decoder,
   and take their difference. Preserve the joint interaction of both moved
   atoms, original electronic states, O(3)/atom symmetry and reversal sign.
   Separate the known restraint change and intrinsic edit volume. A cached
   passive representation must not leak active coordinates. Do not claim global
   cycle consistency from pair reversal alone.
5. Fit parent-balanced objectives, with linear bond-energy and graph-only
   learned controls, then test REAL selected edits promptly. A fixed informed
   balancing rule may turn predicted work into probabilities over the actual
   valid catalogue; the reverse catalogue and normalized probability must be
   recomputed at the candidate. Keep uniform and current-force informed controls,
   as well as bare edit/root noise. Do not present standard locally balanced
   MCMC as a new general algorithm.
6. Both12-parent and18-parent cohorts are now development/evaluation data;
   never fit on them. The18 parents remain evaluation-only even though their
   original generator stream was called fresh_training. Keep722 reserved
   outcomes unqueried. No optimizer recovery is required before this AI test.

All present physical jobs are terminal:46227985/46228322 catalogue and audit;
46225291/46225631 simple controls and audit;46223507/46223792 transfer and audit.
Re-query Slurm before recovery; never restart merely from an expired observation.
The old26-page manuscript and mobility claim need later rewriting around an
actually supported method; do not spend the next turn on PDF formatting.
