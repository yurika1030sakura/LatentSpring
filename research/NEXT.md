# Next research decision

The ICLR objective remains active and unachieved. No replicated new molecular
contribution is established. Do not restart completed audits or resubmit live jobs.

Current 300-K experiments (source 71a853f; each 500 updates, batch 16, 16 path steps,
256 evaluation samples, seed 9051, 8,512 potential queries, maximum 2 GPU-hours):

- 45744181: work300_fixed_joint_5846_v1, fixed noise, joint work training.
- 45744274: work300_annealed_joint_5846_v1, square-root noise, joint work.
- 45744278: work300_annealed_energy_5846_v1, same annealing, energy-gradient control.
- 45744280: work300_annealed_joint_1137_v1, AgBr2, annealed joint work.

All four were RUNNING at September 9 22:34 UTC. Their completed-result assessments
are queued from source f8a61d9:

- 45745430: work300_assessment_5846_v1, afterok on the three eight-atom runs.
  Compares final samples with explicit fixed_joint/annealed_joint/annealed_energy
  labels and the same 32 fixed xTB indices; all 256 geometries enter diagnostics.
- 45745431: work300_assessment_1137_v1, afterok on 45744280. Compares initial/final
  normalizers and invariant moments with the independent reference, then xTB.
  Empirical importance-sampling errors cannot reveal a commonly missed tail.

Next actions:

1. Inspect exact terminal states and result files for those six jobs. Preserve
   all outcomes. Compare against the existing 8,512-query 300-K FM+MALA baseline:
   xTB 31/32, median successful strain 2.40954 eV, 30 improvements/1 worsening/1
   both-failed against the initial FM panel. These are MALA gains, not novelty
   or proof of equilibrium. The 1000-K control gives 30/32 and 2.07368 eV.
2. Apply the terminal-noise diagnosis carefully. Gaussian smoothing implies a
   necessary target-curvature ceiling 40.050 eV/A2 for the old fixed schedule;
   four finite-force probes show 185--233 eV/A2. Square-root noise raises the
   ceiling to 640.050, but passing this local screen does not prove accuracy.
   This uses established second-order Tweedie mathematics, not a new theorem.
   See notes/terminal_noise_resolution.md. Both initializations retain the same
   pretrained field by neutralizing the old constant-temperature input column.
3. Independent AgBr2 cold reference 45738454 completed: four 4,096-point scrambles,
   relative normalizer SE 4.18% at 300 K and 1.45% at 1000 K; minimum ESS 116.6/632.2.
   New queries 16,408 plus the separately retained 12,304-query pilot. No learned
   FM templates enter the proposal. The full 6D Jacobian and Gaussian integrals
   pass. This is statistical reference evidence, not certified global coverage.
   Refine if candidate accuracy reaches the current reference uncertainty.
4. Expand a supported comparison across training seeds and the 664 audited new
   development conditions. The 722 reserved conditions remain free of method
   outcomes until a protocol is frozen. Condition-only input passes a real
   PbCl2 smoke without opening reference-coordinate data.
5. Require practical value beyond SNF/FEAT, FKC, EWFM, FALCON, RegFlow and direct
   MCMC. Equivariant augmented coupling flows are also established (NeurIPS 2023).
   Neither corrected identities, reference construction nor noise scheduling
   alone establishes the contribution. Distill only a qualified teacher.

Retained 1-eV controls correspond to about 11,604.5 K. Independent AgBr2 quadrature
is extended; compactness/xTB quality are not direct correctness tests for that
hot target. Reference-mean 500 joint: ESS 6.894/256, xTB 12/32, median 15.8157 eV.
Native-mean 500 joint: ESS 8.671/256, xTB 16/32, median 15.3193 eV. Both energy-only
controls have xTB 32/32 but severely concentrated weights. The direct 1-eV MALA
control worsens structure quality. Keep all of this evidence. The new 300-K and
1000-K targets still include an explicit .1-eV/A2 restraint and are not unconfined
or empirically inferred OMol thermal ensembles.

Data audit is complete: 3,941,522 accepted raw/processed records match bitwise;
original charge, spin, sources and float64 energy are restored. Use immutable
source_index_readonly.sqlite. Official validation: 2,762,021 total, 2,564,135
eligible; explicit source links do not exhaust all parent-trajectory relations.

209 tests pass. Last main-text build: 9/9 pages, still an audit/development draft.
Code/tests are not ICLR readiness. Authors and submission belong to the user.
No subagents or external messages are authorized. Never write home or edit the
shared FlowMol installation. Source hashes and every submission are recorded.
