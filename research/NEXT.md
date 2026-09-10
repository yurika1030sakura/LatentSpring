# Next research decision

Goal active, unachieved. 228 tests pass. All three pair-precision training arms
45780761/45780762/45780775 and xTB45781150 have completed. No current broad
sampling advantage or validated AI novelty; do not present this candidate as
an ICLR breakthrough. Current branch outputs remain in laboratory storage.

1. Final candidate result: learned/fixed/trace ESS1.130/1.230/1.000 of256,
   xTB32/32,31/32,32/32, median successful strain .77571/1.00995/.88126 eV.
   Original isotropic mean-work at8512 queries gives .56999 eV; same-query HMC
   .90211 eV with correlated draws. HMC.56135 eV uses25024 queries, not8512.
   Evidence: pair_precision_full_comparison_v1.json. Do not extend this recipe
   or scale it across conditions without resolving its failure.
2. Frozen precision diagnostics on first32 final geometries at t15/16 find
   learned max eigenvalue1.03366 and trace1.00198 versus fixed6.15194. This
   suggests weakening near the endpoint; it does not establish full-path
   behavior or a cause. If needed, inspect actual retained trajectories rather
   than interpreting final geometries as preterminal states.
3. Next scientific diagnosis: separate poor forward endpoint coverage from
   auxiliary reverse-kernel/weight variance. A frozen-forward experiment can
   change backward modelling while keeping generated coordinates fixed.
   Audit weight-moment conditions first: positive bounded covariance is not
   itself a finite-importance-variance guarantee. State any proposed gradient
   or tail mechanism as a hypothesis until verified. Do not add another long
   covariance-training variant solely because its training loss declines.
4. Previous1500-step isotropic continuations also failed calibration despite
   better geometry. Their complete audit is work300_5846_1500_comparison.json.
   All earlier LV, density, SMC and pilot failures remain relevant.
5. AgBr2 has repeated weighted checks and an independent same-box cubature
   cross-check. It is only one three-atom condition, not general calibration.
   The four32-particle HMC-SMC references remain high-variance failures.
6. New eight-condition development baseline is complete. Candidate physics
   outcomes are untested; reserved722 conditions untouched. Final ICLR claims
   still require a distinct learning contribution, multiple conditions/seeds,
   reliable distribution checks and honest full-compute comparisons.

See PAPER_STORY_CURRENT.md and STATUS.md. Preserve all raw outputs and source
snapshots. Never write home or alter shared FlowMol. Keep OMol25 bond-free,
max_atoms200, actual charge/spin, and the two environments separate. No goal
completion or genuine external blocker is established by these failures.
