# Next research decision

Scientific goal unachieved.253 tests pass. Latest residual-calibration jobs
45820074/45820248/45820530 all completed. No molecular actor update, no established
new-method sampling advantage, and no external blocker.

1. A concrete candidate is specified in notes/stein_calibrated_entropy_candidate.md:
   learn the actual proposal score, fit a conservative residual correction on
   separate samples, qualify it independently, then combine its entropy gradient
   with the physical target force. The frozen calibration layer is implemented;
   the molecular actor loop is not yet qualified or run.
2. Version1 has a limited positive result: both neural seeds improve heldout
   Stein estimates of score risk and beat calibrated Gaussian DSM. But both fail
   the complete gate: fitted moments4/4, unfitted moments2/10 and4/10, with large
   radial/angular violations. Keep all results in score_calibration_audit_v1.json.
   Do not call it a calibrated generator or extend the same four directions.
3. The concrete next candidate is a richer learned invariant feature correction,
   e.g. a convex refit of the invariant energy critic's linear head. It is not
   implemented yet. Separate representation error from optimization error and
   compare against strong fixed-feature Stein/score-matching baselines. Gaussian
   conditioning, Stein projection, score matching, VSD/DMD and NDSM are prior art;
   no identity or new name alone supplies novelty.
4. The16384-parent panel (seed9117, noise9118) is independent evidence forversion1.
   After examining it to design a new variant, treat it as development data and
   generate a fresh confirmation panel before promotion. Never fit the failed
   probes and present their training fit as validation. Reserved722 conditions
   remain untouched; candidate outcomes on the frozen eight-condition panel
   are still untested. AgBr2 evidence remains limited to three atoms.
5. Preserve all earlier failures: joint work, covariance, LV, SMC, CNF numerics,
   empirical CFM, global Gaussian auxiliary and neural score qualification.
   The four-direction correction does not erase them or establish ICLR readiness.

Never write home, alter shared FlowMol or merge environments. Keep original
charge/spin, bond-free OMol25 and max_atoms200. Current manuscript is a research
and audit draft. Goal completion and a genuine external impasse are not established.
