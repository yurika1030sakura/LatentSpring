# Next research decision

The ICLR objective is active and unachieved. 217 tests pass. Use STATUS for the
full result history and failures. Latest new source ba5cc0d; all job source hashes
are recorded in jobs.jsonl. Do not resubmit live jobs.

1. Inspect eight-atom fixed-observation log-variance training 45760898, output
   logvariance_300K_5846_500_v1, source 331238b. It uses 500 updates, B16, K16,
   native mean, noise power .5, kT .025851999786435, seed 9051; 256 before/after
   evaluations and 8,512 total oracle calls. Both production smokes passed,
   including mixed-time likelihood/gradient agreement; performance unproven.
   Compare to completed same-target joint and energy arms, retaining their ESS~1.
   xTB assessment 45761719 waits on afterok:45760898 and retains all 32 attempts.
2. Inspect eight-atom HMC 45761145 (hmc_300K_5846_v1). Same 8,512 queries and
   unchanged AgBr2 HMC recipe; no eight-atom normalizer reference. Keep eight
   chain clusters, all initial starts and all relaxation failures visible.
3. AgBr2 fixed-joint 45750959 and energy-only 45750966 remain live. Their paired
   4096-path evaluations for seed 9065: 45759434/45759436; second independent
   stream 9067: 45760430/45760432. These depend on successful training completion.
   Inspect all controls before concluding noise annealing or entropy benefits.
4. All three annealed-joint seeds and both independent evaluation streams have
   completed. Combined ESS 802.65/823.79/640.28 of 8192, log-normalizer differences
   -.05450/-.03472/-.06264 nat, empirical relative errors 3.35/3.30/3.79% plus
   shared reference 1.82%. Preserve individual streams; never select closest to
   reference. No advantage over HMC or broad calibrated sampling is established.
5. New eight-condition development FM/xTB panel completed: 512 generated attempts
   plus eight references, FM64 xTB 239/256. No physics training on this panel yet;
   reserved 722 conditions untouched. Resolve the eight-atom failure before scale-up.

Log-variance path training is established prior art, not new theory. Its observation
measure is fresh current-forward paths held fixed within each gradient update;
no replay or endpoint force gradient. The old mean-work objective remains default.
Target is eSEN plus an explicit COM restraint, not an unconfined DFT ensemble.
Last paper build: nine main pages, no undefined references/citations. Scientific
submission readiness false. Author/submission tasks remain with the user.
Never write home, alter shared FlowMol or restart live jobs to shorten waiting.
