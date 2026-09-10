# Next research decision

Scientific goal unachieved. All pair-precision and frozen-backward-refit jobs
are complete; no general sampling gain or validated AI novelty is established.
The persisted goal tracker is paused; user has explicitly requested continued
work in the current turn. No external blocker has been identified.

1. Completed independent4096-path forward teacher export45811812, seed9084.
   Raw ESS2.625/4096; stabilized ESS2048 is a training control only.
   Source is annealed-joint1500 on condition5846, independently refitted backward
   model. Keep full path weights and all oracle/source costs. A low raw ESS is
   a limitation, never concealed by reporting stabilized training ESS instead.
2. CFM student code and real-GPU smoke45812206 pass.237 tests pass.
   Source45812380 and uniform45812384 are running. Linear/power initial submits
   were rejected by gpu_test two-job QOS; preserve rejected v1 directories and
   submit v2 after slots release. No dependent audit submitted yet.
   The prespecified uniform/linear/power CFM student comparison uses
   identical initialization, training budget and fresh independent Gaussian
   starts. Evaluate the full original target on independent generated paths;
   measure projection/sampler mismatch rather than assuming it vanishes.
3. The fixed-forward backward refit45802060 failed: all four variants ESS near
   1/256. Evidence is backward_refit_5846_v1.json. The forward samples did not
   change; this only tests the same backward family and scalar widths.
4. All pair-precision arms failed (ESS1.13/1.23/1.00 of256). Do not extend them.
   Same8512-query HMC strain is .9021 eV; .5613 uses25024 queries. All failed
   LV, SMC, density and1500-step work results remain part of the audit.
5. AgBr2 repeated weighted checks and independent same-box cubature remain
   limited to three atoms. The eight-condition development baseline is ready;
   candidate outcomes and reserved722-condition outcomes are untested.
6. Before any ICLR claim require a distinct learning contribution, multi-condition
   replication, credible distribution validation and full-budget comparisons.

See STATUS.md, PAPER_STORY_CURRENT.md and notes/forward_mass_update_candidate.md.
Keep immutable source snapshots, actual charge/spin, bond-free OMol25 and
max_atoms200. Never write home or alter shared FlowMol/environment installations.
