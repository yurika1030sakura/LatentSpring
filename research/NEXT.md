# Next research decision

The ICLR objective is active and unachieved. Current source ae63de3, 216 tests
pass. See STATUS for historical evidence and failed branches; preserve them.

1. Inspect completed long MALA/HMC controls 45759386/45759394, each 8,512 queries.
   Eight independent chains supply 64 correlated draws; use clustered errors.
   Do not interpret target-invariant transitions as convergence or endpoint density.
2. Inspect fresh AgBr2 4096-path evaluations: 45759407 (9051), 45759409 (9052),
   45759411 (9053), 45759434 (fixed joint, afterok:45750959), 45759436 (energy,
   afterok:45750966). All use v2 reference, same seed 9065/batch 64 and separate
   evaluation cost. Training controls remain live. Do not select a winning seed.
3. Fixed-path scoring core is tested analytically. Validate time-batched values
   and parameter gradients against sequential real FlowMol before molecular use.
   Log-variance path-space training is prior art; specify the observation measure
   and detached states. This is a candidate baseline, not demonstrated novelty.
4. New eight-condition development FM panel has completed. All 512 attempts and
   eight reference relaxations are retained; FM64 xTB succeeds 239/256. Physics
   benefit across this panel remains to be tested. Reserved 722 conditions untouched.
5. AgBr2 three-seed ESS is 44.08/51.20/20.88 of 256. Independent v2 normalizer
   uncertainty is 1.82%; candidate errors remain larger. Eight-atom joint/energy
   training still yields ESS near one despite improved relaxed geometry. Resolving
   this failure and proving matched-baseline benefit are scientific gates.

The potential and explicit restraint define the target. Temperature/noise changes
are not standalone novelty. Last paper build is a 9-page development draft;
scientific submission readiness remains false. Authors/submission are the user's.
Never write home, modify shared FlowMol, or restart live jobs to shorten waiting.
