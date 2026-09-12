# Claude continuation — September 12, 2026

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `CLAUDE.md`, `research/NEXT.md`, `research/STATUS.md` and
`research/DELAYED_SCREEN_STATE_20260912.json`. Keep environments separate and
never write to the home checkout.

The four delayed-screen training arms and independent audits are complete.
They use the fixed physical site-arc proposal and exact two-stage correction.
Internal learned query-rate gains over no screen are 31% / 43%, while the fixed
physical control's point gain is 93%. Neural and four-parameter linear results
are nearly identical; their difference intervals span zero. No useful neural
advantage, achieved query saving, full-chain or wall-time benefit is established.
Do not scale those neural weights. All prior negatives remain.

Summary: `runs/delayed_screen_summary_v2/results.json`; v1 is retained. Every
model/control metric and 6,388 supported-pair cases are independently checked,
with both orientations, factor balance and costs. Maximum discrepancy is
3.09e-14; six trained-head finite differences have maximum error 4.93e-10.
Jobs 46196124 and 46196212 are terminal; re-query Slurm before action.

The next bounded candidate may use the source force already cached by the
sampler. `runs/screen_force_pairs_v1` contains all 1,671 old pairs/failures and
unchanged splits, with 3,233 used states' forces reconstructed exactly from raw
and inverted oracle forces. No new physical calls were used. Candidate force
is available only after its query: it must never be fed into a forward screen.
Read `notes/cached_force_screen_candidate_v1.md` and NEXT. Source-force gates
need the general reverse/forward gate correction, not an assumed antisymmetric
R-s formula. No force-screen model or protocol is implemented or frozen yet.

The tensor EnergyOracle's buffered noisy-output timeout was reproduced with a
fake worker and fixed using the existing NumPy oracle's byte-buffer reader.
Values and requested/acknowledged counts still pass tests; RPC wall time is now
recorded separately. This does not retroactively time or invalidate prior runs.

Keep only 36 FIT parents in optimization; retain the 12 internal-selection
parents and all failures. Never fit on fresh follow-up outcomes, either prior
molecular evaluation cohort, or the 722 reserved conditions. Inherited data
construction costs remain 18,110 raw calls. The ICLR goal remains active and the
paper scientifically unready; authors and actual submission remain with the user.
Prior handoff: `notes/archive/claude_handoff_through_action_geometry_complete_20260912.md`.

Verified development PDF: `runs/verification/delayed_screen_completed_20260912/main.pdf`
(9 main pages, 22 total). Build record:
`research/evidence/delayed_screen_completed_build_20260912.json`. This does not
establish scientific submission readiness.
