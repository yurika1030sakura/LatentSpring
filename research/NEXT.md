# Next research decision

Goal remains active and unachieved. Latest scientific status: September 10 UTC.
Last full suite: 222 tests pass; two new quadrature/noise-limit tests also pass.
Latest manuscript build has 9 main pages, no unresolved references.
No broad calibrated molecular benefit or ICLR-ready contribution is established.

1. Live continuations to 1500 total steps (start from 500; optimizer and global
   RNG step restored, no second temperature reset):
   - 45763360 work300_fixed_joint_5846_1500_v1
   - 45763361 work300_annealed_joint_5846_1500_v1
   - 45763362 work300_annealed_energy_5846_1500_v1
   Source 4b4c615. Each adds 16,512 queries, cumulative 25,024 including earlier
   8,512. Assess final ESS, energy, work variance and geometry only when complete.
   xTB 45764564 depends on all three and retains 32 attempts per arm.
   If a run fails, preserve failure and assess completed arms separately.
2. Matched 25,024-query HMC 45763216 completed: xTB 32/32, median strain .56135
   eV. Eight chains supply 64 correlated draws; no normalizer/mixing certificate.
3. Independent box integral order 48 is live as 45770866. Orders 20/32 of
   completed 45767649 differ from same-box QMC by +.04961/+.005303 nat. Inspect
   order stability before refining numerical claims; finite box is not full target.
   All four HMC-SMC seeds completed with pooled Z/reference 1.723, descriptive
   SE .528. None is selected for promotion.
   HMC-SMC reference 45764408 failed its predeclared AgBr2 screen: log Z error
   +.7134 nat versus .25 threshold, despite ESS 23.14/32 and distance error .00945 A.
   Five ancestors have weighted ancestry ESS 1.326. Do not declare this a reference
   or extend it directly to eight atoms. Any redesign requires a new explicit
   protocol; do not relax the existing gate. Source 73cee25, 32,832 oracle calls.
4. Completed 500-step log-variance recipe fails: ESS 1/256, overlap 99.22%, xTB
   25/32, median successful strain 316.30 eV. Preserve it; no scale-up. Frozen
   gradient diagnostic 45762271 measured 37.6-fold forward covariance trace
   increase after population-scale correction. Backward objectives also differ;
   a unique failure mechanism is not established. NeurIPS 2025 arXiv:2506.10982
   already distinguishes LV and reverse-KL bridge training.
5. All AgBr2 five arms and two independent evaluation streams completed. Joint
   annealed seeds have ESS 802.65/823.79/640.28 per8192; fixed joint 886.45;
   energy-gradient-only 13.72. Their log-Z differences are -.05450/-.03472/-.06264,
   -.03239 and -5.06827 nat. No annealing advantage over fixed noise is established.
   Reproduce using combine_work_evaluations.py and input paths in
   research/evidence/work300_all_triatomic_controls_v1.json. Keep both streams.
6. New eight-condition development FM/xTB baseline is complete, all 512 generated
   attempts plus eight references retained; FM64 xTB 239/256. Physics-training
   outcomes on this panel remain untested. Reserved 722 conditions untouched.
   Resolve larger-condition sampling before expanding or making transfer claims.

Current source/data/target invariants remain in CLAUDE.md. Never write home or
modify shared FlowMol. Every submission uses a committed immutable snapshot.
Do not restart live jobs or equate engineering PASS with scientific qualification.
Authors and submission remain with the user. Keep all failures and old outputs.

The actual compute ledger and Gaussian surrogate limit are in STATUS. Neither
matched oracle queries nor local harmonic calculations establish molecular
sampling efficiency. Do not add architectures while the live continuations
and independent reference check can resolve the current decisions.
