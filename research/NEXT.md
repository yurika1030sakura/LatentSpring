# Next research decision

The ICLR objective is active and unachieved. One independent three-atom
calibration is promising; replicated advantage over strong baselines and larger
conditions remains unproven. Do not restart completed audits or resubmit live jobs.

Current work:

1. AgBr2 replication/controls (source be9fb7a, 300 K, 500 updates, batch 16,
   16 transitions, 256 evaluation samples, 8,512 potential queries each):
   - 45750942: work300_annealed_joint_1137_s9052_v1, RUNNING at 23:36 UTC.
   - 45750951: work300_annealed_joint_1137_s9053_v1, RUNNING.
   - 45750959: work300_fixed_joint_1137_v1, PENDING for priority.
   - 45750966: work300_annealed_energy_1137_v1, PENDING for priority.
   Inspect terminal states and completed outputs, then evaluate all arms with
   the same independent stream. Do not select the best seed or omit a control.
2. Frozen loader validation 45751790 (work300_loader_validation_1137_v1,
   source 1d7da56) is PENDING. It must reproduce 32 saved evaluation paths at the
   original batch 16; inspect source_stream_comparison before using larger runs.
   Inference loads both saved networks and the actual target-kT input. The
   initialization temperature reset is already baked into trained weights and
   must not be applied again. Once validated, run 4096 independent samples per
   completed proposal with seed 9065 / batch 64, accounting evaluation cost separately.
3. Pure AgBr2 FM control 45747807 (work_fm_control_1137_v1) waits in the regular
   GPU queue; its training dependency 45744280 is complete. It generates 64 FM
   samples at the checkpoint's trained input-kT, not an untrained cold input.
   MALA 45749291 (fm_mala_300K_1137_v1) depends on it and includes an independent
   moment comparison. MALA endpoint density, importance ESS and log Z remain
   unknown. Its finite-time sampling error does not include initialization bias.
4. New development panel: frozen candidate indices 194,224,147,169,120,29,109,53;
   9--20 atoms, eight size/charge/spin strata. Raw references are checked and
   the condition-only CPU interface succeeds 8/8. Full baseline 45747806
   (development_fm_baseline_v1) is PENDING for priority; assessment 45749290
   (development_fm_assessment_v1) depends on it. Retain all 512 generated-output
   attempts plus eight references, including generation and xTB failures as
   distinct counts. No reference coordinates initialize generation. See
   notes/development_panel_protocol.md. Reserved data remains untouched.

Completed results:

- AgBr2, 300 K, annealed joint seed 9051: ESS 44.080/256, maximum weight .06089.
  Log-normalizer difference to independent reference is -.00924 nat, with
  empirical relative SE 13.73% versus reference 4.18%. Weighted moments agree
  within estimated error. xTB 32/32, median strain .45196 eV (initial 28/32, 1.61084).
  One condition/seed and finite-reference uncertainty do not establish ICLR value.
- Eight atoms, 300 K: fixed joint / annealed joint / annealed energy all xTB 32/32,
  median strain .56999 / .94594 / .13002 eV, but ESS 1.0003 / 1.0687 / 1.0006 of 256.
  Geometry improvement is not target calibration. Keep this failure visible.
- Same eight-atom FM+MALA, 8,512 queries: 300 K xTB 31/32, median 2.40954 eV;
  1000 K 30/32, 2.07368 eV. Neither finite chain is declared equilibrated.
- Independent AgBr2 cold reference 45738454: four 4096-point scrambles,
  normalizer relative SE 4.18% at 300 K and 1.45% at 1000 K. New queries 16,408;
  independent pilot 12,304. Refine if candidate uncertainty reaches this level.

Keep target definitions explicit: retained 1-eV controls are about 11,604.5 K;
compactness/xTB quality is not a direct calibration test of that extended target.
Physical-temperature targets still include the explicit .1-eV/A2 restraint.
The Gaussian smoothing curvature restriction uses established Tweedie theory;
square-root noise resolves the measured necessary local screen, not all sampling
problems. Coupling flows, SNF/FEAT, FKC, EWFM, FALCON and RegFlow are prior work.
No standalone identity or architecture repair is sufficient novelty.

Data audit is complete: 3,941,522 raw/processed records matched bitwise, original
states/sources/float64 energies restored. Use immutable source_index_readonly.sqlite.
Official validation has 2,762,021 records and 2,564,135 eligible; explicit source
links do not exhaust every parent-trajectory relation. The 722 reserved conditions
have no method outcomes. Do not reuse the duplicated legacy test as a blind test.

210 tests pass. The last main build is 9/9 pages and remains an audit/development
draft. Authors/submission belong to the user. No subagents or external messages
are authorized. Never write home or modify the shared FlowMol installation.
