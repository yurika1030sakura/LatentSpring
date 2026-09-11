# Next research decision — 2026-09-11

The ICLR goal remains active and unachieved. This turn made progress: matched
physical MALA/HMC controls, all-proposal replays, independent geometry comparisons,
and the corresponding manuscript update are complete. No submission-ready
method or calibrated molecular sampler is established.

## Latest evidence and decision

- Four physical MALA/HMC replicas are COMPLETE and replay all2048 parents and
  every proposal. Each used18048 raw queries, including both orientations and
  rejections; two176-query engineering cases give72544 total new raw calls.
  MALA jobs45973718/45974132 use60/64 seconds of whole allocation; HMC
  jobs45974116/45975150 use65/64. Mean projected-energy reductions are
  4.1013/4.1189 and4.1998/4.2050 eV. These are NOT KL changes.
- Their independent geometry job45975404 completes128 attempts,119 converged,
  9 failed. MALA medians3.0524/2.8456 eV with31/29 of32 converged; HMC
  2.7689/2.5736 eV with30/29. Current convex is4.6595/4.6550 eV with28/29.
  All480 condition0 attempts across15 arms retain423 converged and57 failed.
  Read evidence/parity_physical_controls_v1.json and
  evidence/parity_all_controls_geometry_v1.json. Short chains are not independent
  equilibrium references; no endpoint density, normalizer or ESS is assigned.
- Full EACF condition0 replicas45960348/45961068: joint KL changes
  -118.9897 +/-6.5106 and -105.5599 +/-6.4687 nat across512 parents each.
  Convex marginal changes are -24.7415 +/-2.2942 and -24.6575 +/-2.2692.
  Each arm uses18048 raw oracle queries. Full EACF has5629656 parameters;
  convex has19365. All1024 EACF checkpoint parents replay exactly.
- Compact EACF replicas45967320/45967327:14900 parameters chosen by target-free
  shape counts; joint changes -27.2252 +/-2.6661 and -32.5646 +/-3.0341 nat.
  Paired differences from convex are -2.4837 +/-1.1921 and -7.9071 +/-1.9567.
  Four/nine guarded optimizer attempts were ignored; all1000 attempts and18048
  queries per arm are retained. Audits45968020/45968027 replay all1024 parents.
- Joint KL change upper-bounds EACF marginal change in expectation. Thus a
  more-negative joint change counts AGAINST convex marginal-quality superiority;
  reverse ranking would not prove convex superiority. Row SEM is not uncertainty
  across training seeds. No endpoint importance weights were evaluated.
- Whole-allocation seconds: convex139/144, compact196/198, full2843/2843.
  Between steps20 and1000, compact uses .080/.082 seconds per attempt, versus
  convex .108/.106. The former20x full-model timing difference is NOT a general
  or equal-quality acceleration claim. Compilation, unequal model capacity,
  achieved quality and physical-host contention remain relevant.
- Independent GFN2 geometry: full EACF29/32 and28/32 converged, success-only
  median strains3.2681/3.8977 eV; convex28/32 and29/32,4.6595/4.6550 eV.
  Compact29/32 and26/32,4.8488/4.6620 eV, with mixed paired rankings. All352
  condition0 assessments across11 arms retain304 converged and48 failed rows.
  Input atom identities/coordinates, source contracts, binary hashes and raw
  successful-energy/convergence logs were independently checked. These metrics
  are not full chemical validity or target-population coverage.

Canonical comparisons:
- evidence/parity_physical_controls_v1.json, with each run independently checked
  by scripts/research/audit_parity_mcmc.py. Source/proposal traces are immutable.
- evidence/parity_all_controls_geometry_v1.json includes the physical controls.
- evidence/eacf_full_compact_comparison_v1.json, produced by
  scripts/research/compare_eacf_controls.py (no new physical queries).
- evidence/eacf_geometry_comparison_v2.json, produced by
  scripts/research/compare_geometry_assessments.py (no new xTB calculations).
- Earlier prefixes and full-only comparisons remain immutable historical evidence.

Decision: the present convex architecture has no established external advantage.
Do not expand its capacity or spend reserved outcomes merely to seek a positive
result. Finish the bounded main campaign, then use matched physical-sampling
controls and credible distribution references to decide what reconstruction is
scientifically justified. The pair-potential capacity probe also improved mean
KL without reliably improving ESS; its capacity-only expansion remains stopped.

## Live main campaign — re-query Slurm before acting

-45914826_5 and45914826_6 are RUNNING, with four-hour caps. Last query had elapsed
  2h40m20s and55m47s respectively. The other six molecular conditions are terminal.
  Do not restart on an observational timeout or duplicate these outputs.
- evidence/parity_production_prefix_v5.json audits39 completed arms,0 failures,
  9 unresolved,703872 acknowledged queries in completed arms. Total planned48
  arms,866304 queries, plus36864 completed source-generation queries and all
  other experiments/failures. Never replace actual counters with planned budgets.
- Geometry for conditions0,1,2,3,4,7 is complete; condition4 job45967727 adds224
  attempts. evidence/parity_geometry_prefix_v2.json records1344 attempts,
  1246 converged and98 failures. Conditions5/6 still require geometry after all
  six training arms in their panel.json have qualified.
- Run roots: condition0 runs/parity_entropy_condition_00_v1; others
  runs/parity_entropy_conditions_1_7_v1/condition_XX. Immutable main training
  source is41516838cc13da4ffce9097404f95e3193422640.
- Use compare_parity_campaign.py on a NEW evidence path for the final campaign.
  Launch parity_geometry_condition.slurm via submit.py for newly complete5/6.
  Do not overwrite any prior training, assessment or evidence prefix.

## Next substantive experiments

1. Finish the above48-arm/8-condition matrix, preserving all seeds and failures.
   Do not claim all-condition success from the current prefix.
2. The exact-query physical control is now qualified and complete in
   parity_mcmc_control.py. Do not repeat its original short recipe merely to
   obtain more favorable results. The old fm_initialized_mala.py/long_chain_mcmc.py
   still use old source schemas and raw targets; they are historical.
3. Establish independent target-distribution references with declared starting
   ensembles and mixing/uncertainty checks. Short chains or acceptance alone do
   not qualify them. The eight audit-linked raw development reference geometries
   are already in runs/development_fm_baseline_v1/references.json. Its manifest
   hash is cd0cff7c0b3422e1ae560b41c346d1753e5ebfc1ad7629387998911bd87f24ad.
   Use them only for independent reference assessment, never generator input.
4. Diagnose useful architectural changes against BOTH full and compact EACF,
   together with physical-sampling controls. Current correctness repairs, KL
   cancellation, convex maps, equivariance and AFM/Jarzynski motivation are not
   sufficient new-method claims. No further unvalidated capacity grid is justified.
   Read notes/physical_control_decision_v1.md: Timewarp and Markovian Flow Matching
   already cover conditional-flow/MH proposals and FM within adaptive MCMC.
   A justified architecture/learning intervention must solve a measured bottleneck;
   merely adding Metropolis correction cannot be promoted as new AI novelty.

## Contracts that must remain fixed

Frozen FM64 midpoint source at T=1, displacement head, .025-A terminal COM noise;
base neural1-eV input is not physical temperature. Work on labelled, unweighted
zero-centroid coordinates with fixed composition/charge/spin. Use inversion-mixture
q0_plus and E_plus=(E_raw(x)+E_raw(-x))/2,
F_plus=(F_raw(x)-F_raw(-x))/2, kT=.025851999786435 eV, restraint=.1 eV/A^2;
U=E_plus+.05 sum||x||^2. The one-raw-query shortcut is valid for the invariant
linear training expectation, never inside weights or MH acceptance. Keep the
source frozen for source-entropy cancellation. Preserve all prior score/CNF/work/
covariance/empirical-CFM/auxiliary failures and the index-split permutation defect.

Reserved722 outcomes remain untouched. Never write home, alter shared FlowMol,
merge the FlowMol/fairchem/JAX environments, or count formatting/tests as scientific
readiness. See CLAUDE.md, notes/claude_update_review_v1.md and STATUS.md for history.

Latest draft: runs/verification/paper_20260911_physical_final/main.pdf,
7 main pages, clean build. It includes the adverse full/compact comparisons and
mixed independent geometry. The paper is not submission ready.
