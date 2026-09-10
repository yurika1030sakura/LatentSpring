# Next research decision

Goal active and unachieved. 228 tests pass. Original manuscript remains an
honest audit/development draft (nine main pages), with no validated ICLR-level
AI contribution or broad calibrated molecular sampling benefit.

1. Live candidate comparison (source ec121f1):
   - 45780761 pair_precision_learned_5846_500_v1
   - 45780762 pair_precision_fixed_5846_500_v1
   - 45780775 pair_precision_trace_5846_500_v1
   Same FM weights, seed 9051, 500 updates/B16/K16, 256 initial/final evaluations,
   300 K and .1 restraint, native means, scalar noise .2, no scalar annealing.
   Each uses 8,512 oracle queries. Precision strength 32, learning rate .001;
   means use 1e-5. Inspect all three, including learned versus fixed identical
   initial sampling, work/energy variance, ESS, geometry and actual cost.
   xTB assessment 45781150 waits for all three to succeed.
   Preserve failed jobs and assess completed arms separately if necessary.
2. Required gates already passed: molecular B2 45779357 (132 queries), B16
   45780203 (160 queries, .443 GiB), loader 45780207 (exact coordinates,
   work max error .000116). Learned precision weights demonstrably updated.
   These are implementation checks, not evidence of a useful new method.
3. The candidate uses learned/fixed pair elastic precision and a same-trace
   isotropic control. Prior art includes elastic networks, AniDS and Chroma.
   An atomwise anisotropic comparator and scaling work remain necessary before
   novelty claims. Current dense cubic factorization and 1/N normalization
   have unproven behavior at larger sizes. See notes/pair_precision_candidate.md.
4. Completed original continuations to 1500 steps fail distributional calibration:
   ESS 1.174/2.218/1.092 of256, all xTB32/32, median strain .49357/.52856/.13984.
   Matched 25,024-query HMC is32/32, .56135 eV, with correlated chains. The
   automatic audit is retained in work300_5846_1500_comparison.json. Do not
   restart or keep extending those recipes solely because work decreases.
5. Independent AgBr2 reference checks are complete. Two evaluation streams of
   three trained seeds plus both controls are retained. Direct box cubature
   orders20/32/48 support same-box QMC normalization; no full-domain certificate.
   Four small HMC-SMC populations are too noisy for promotion to a reference.
6. Eight-condition independent development FM/xTB baseline is complete; no
   candidate physics outcomes on that panel yet. Reserved722 conditions remain
   untouched. A broad final claim needs independent conditions, replicated
   training benefit and fair compute comparisons, beyond the present one-case pilot.

Current paper story: PAPER_STORY_CURRENT.md. Keep inherited audit and all
negative results. Never write home or modify shared FlowMol. Preserve bond-free
OMol25, max_atoms200, true electronic state and separate environments. No new
sampler/AI superiority claim follows from engineering PASS or a positive toy.
