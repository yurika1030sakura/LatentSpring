# Next research decision

The ICLR objective is active, not achieved. Existing results do not establish
both useful molecular generation and calibrated sampling. No replicated new
contribution has been demonstrated. Do not restart completed data audits.

1. Inspect the native-mean 500-update pair: 45732136 (joint), 45732178
   (energy-gradient), source c37c8d9. Both were RUNNING at September 9 21:20 UTC.
   Their independent xTB assessment 45732403 is queued afterok on both. Each
   learning arm uses 8,512 potential queries. Compare completed results, not
   training loss, and retain every failure. Never resubmit these accepted jobs.
2. Inspect the prospective 300-K/1000-K FM-initialized MALA diagnostic after its
   terminal state. Same 64 FM16 geometries, state, .1 restraint, seed 9061,
   132 moves and 8,512 queries. Proposal std is .1*sqrt(kT/1eV) and score cap is
   100*(1eV/kT), retaining fixed capped physical-force drift. All 1-eV results
   stay in the record. See notes/target_temperature_audit.md before interpreting
   geometry changes. Finite-time MALA is not declared equilibrated.
3. Separate generation relevance from correctness for the declared target.
   kT=1 eV is about 11,604.5 K; independent three-atom quadrature itself is
   extended. A higher xTB success rate alone does not validate eSEN Boltzmann
   sampling, and a lower rate alone does not invalidate it. The 300/1000-K
   targets still include an explicit harmonic restraint. OMol has no thermal
   label assumed here; the FM initialization retains its trained constant-kT
   input rather than extrapolating an untrained conditioning feature.
4. Expand only a justified target/method comparison across training seeds and
   the 664 audited new development conditions. The 722 reserved conditions
   remain free of method outcomes until a protocol is frozen. Condition-only
   graph input passes a real PbCl2 smoke without reference-coordinate loading.
   Its edge ordering respects FlowMol's upper/reverse-pair convention.
5. Demonstrate value beyond current SNF/FEAT, FKC, EWFM, FALCON, RegFlow and
   direct MCMC baselines. Work identities, symmetry averaging, defensive IS and
   invertible regression alone are established methods. Distill to an FM student
   only after a teacher's target accuracy and utility are supported.

Completed comparison (condition 5846, one seed):
- Pure FM16/FM64: xTB 29/32 and 30/32; median successful strain 4.76794/4.67530
  eV. One overlap per 64. Maximum 16-to-64 coordinate drift .66522 A, so no
  solver-convergence certificate. FM importance weights remain unknown.
- Reference-mean initialization/100 updates: xTB 0/32 and 6/32; final ESS3.485/64.
- Reference-mean 500 joint: ESS6.894/256, xTB12/32, median strain15.8157 eV.
  Energy-gradient: ESS1.00036/256, xTB32/32, median6.76552 eV; one weight .999822.
  Independent assessment45729911 is complete. Energy improves31 paired cases,
  worsens1. Neither establishes adequate geometry and calibrated sampling.
- Native-mean two-update preflight: initial xTB30/32, median16.7263 eV,
  overlaps21/64, ESS1.928/64. Final28/32, median16.9588, overlaps20/64, ESS1.781.
- FM-initialized MALA at1eV (45733994): complete, 8,512 queries; acceptance.7139,
  xTB25/32, median9.58024 eV. Energy increases by4.122 eV during the fixed run.
  This suggests target-temperature relevance needs scrutiny; it is not proof
  of equilibrium. The source reference passes xTB with strain.89014 eV.

Raw replay is complete: all3,941,522 accepted tensors matched bitwise; original
charge/spin/source IDs and float64 energies are restored. Use immutable
source_index_readonly.sqlite; old WAL failure did not invalidate the producer.
Official validation audit:2,762,021 records,2,564,135 eligible; explicit source
links do not exhaust every parent-trajectory relation. Reserved data is untouched.

193 tests pass. The last main-text build is9/9 pages and remains an audit draft.
These engineering checks are not ICLR readiness. Latest summary figures are in
research/figures/work_campaign_reference500. Source hashes and all outcomes are
in research/evidence; jobs.jsonl records both accepted/rejected submissions.
Authors/submission belong to the user. No subagents or external messages are
authorized. Never write home or modify the shared FlowMol installation.
