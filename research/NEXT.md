# Next — main-generator symmetry/collision pairing pilot

CURRENT: `research/ORBIT_PAIRING_STATE_20260913.json` and
`notes/orbit_pairing_fm_v1.md`. The ICLR goal remains active and unachieved.
The previous mass-calibration and molecular-routing branches are preserved;
neither is a qualified main-method success. Do not restart their sweeps.

Active jobs (query Slurm; these are snapshots):
-46323226: `runs/orbit_pairing_train_v3`, three sequential FM continuations,
 independent / rotation / steric,3000 steps each, identical initialization/data.
-46323424: `runs/orbit_pairing_eval_v2`, after training;2048 fresh molecular
 outputs, including the frozen warm generator and all8 development conditions.
-46324210: `runs/orbit_pairing_audit_v1`, after evaluation; replay every structural
 outcome and verify matched training records, checkpoint hashes and sample streams.

All three fit arms preserve the Gaussian source via a shared independent Haar
rotation of both training endpoints. Standard rotation alignment is prior art.
The steric arm additionally searches a fixed set of13 proper rotations using
path overlap and displacement. It uses no new oracle, bond labels, generated
evaluation parents or reserved outcomes. Its usefulness is unknown until the
fresh generation comparison completes. Do not equate lower training loss or
path overlap with better output validity.

Finish the current experiment:
1. Verify all three training completions, then the32 method/condition output
 rows and the structural audit. Preserve failed jobs or samples rather than
 replacing a condition. Do not tune weights, angle candidates or training length
 using evaluation outcomes.
2. Run `scripts/research/summarize_orbit_pairing.py` with:
 `--protocol research/evidence/orbit_pairing_protocol_v1.json`
 `--run runs/orbit_pairing_eval_v2/evaluation`
 `--audit runs/orbit_pairing_audit_v1/audit.json`
 `--out research/evidence/orbit_pairing_summary_v1.json`
 `--csv research/evidence/orbit_pairing_table_v1.csv`.
3. If the audited pooled graph-supported count for steric exceeds BOTH
 independent and rotation, perform the already frozen all-output energy check:
 all four methods, all eight conditions, same64 outputs each; E_plus=(E(x)+E(-x))/2,
 original charge/spin,0.1-eV/A^2 COM restraint,4096 raw calls total. Retain invalid
 outputs and errors in the denominator. Otherwise stop this bounded pilot with
 zero new molecular oracle calls. This gate is a feasibility screen, not a
 significance criterion or ICLR-readiness certificate.
4. A positive generation result still needs training-seed confirmation and a
 careful contribution statement against equivariant FM, ET-Flow, SemlaFlow and
 physics-aware paths. These controls are deliberately required before calling
 this an AI contribution. No all-case perfection or new physics law is required.

The old independent-Gaussian velocity-to-score formula is incompatible with
correlated endpoint pairings. This pilot uses FM only; the existing BGFM hook and
three-term interface remain intact. Do not silently enable that force proxy.
The finite T1 midpoint64 sampler plus0.025-A COM noise has no qualified absolute
q, so no importance weights, ESS or Boltzmann-sampling claim follows here.

Scheduling provenance:46322261 and46322800 were cancelled while still PENDING,
with no training metrics or calculations. Partition-update attempts were rejected
by site validators; the subsequent three-element gpu_test array submission was
rejected by the two-submitted-job QOS limit. The accepted replacement uses one
serial training allocation and one dependent evaluation. All records remain in
`research/evidence/orbit_pairing_queue_update_v1.json` and `research/jobs.jsonl`.
Do not cancel other projects' jobs to free this QOS.

Protected:722 reserved outcomes unqueried;12/18 evaluated generated-parent
cohorts excluded from fitting; separate FlowMol/oracle environments; no home writes;
OMol25 primary, bond supervision zero, max_atoms200. User handles authorship and
submission. `paper/angular_working.tex` is still the old diagnostic draft.
