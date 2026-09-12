# Next research actions — September 12, 2026

Read `research/DELAYED_SCREEN_STATE_20260912.json`. Use the laboratory checkout;
never write to home. The full ICLR goal remains active and scientifically unachieved.

1. The four learned delayed-screen arms and independent audits are COMPLETE.
   Summary: `runs/delayed_screen_summary_v2/results.json` (v1 is retained).
   Learned internal rates improve 31% / 43% over no screening, but the fixed
   physical control has a higher 93% point improvement. Neural versus linear
   intervals span zero. No achieved savings or whole-chain gain follows. Do not
   scale the neural weights or claim a new AI contribution from this result.
2. Read `notes/cached_force_screen_candidate_v1.md`. The force-augmented pair
   dataset is ready in `runs/screen_force_pairs_v1`: all 1,671 attempts, original
   36 FIT / 12 internal-selection split, 3,233 force states reconstructed exactly
   from raw/inverted queries. Source force is already cached sampler state.
   Candidate force is unavailable before querying it and must never enter the
   forward screen. No new oracle calls were used; no force-screen model is trained.
3. First implement and test the GENERAL screened-proposal correction:
   log alpha_2 = min(0, R + log g_reverse - log g_forward). A source-only force
   work estimate is not antisymmetric; do not reuse R-s without qualification.
   Test finite-state balance, actual molecular reversal, zero-screen RNG/query
   equivalence, pre-query information access and paid-query error accounting.
4. Before fitting, freeze a small comparison with no screening, the completed
   fixed physical screen, a source-work physical control, a simple fitted screen
   and any neural addition. Keep the base proposal and move-family schedule.
   Add constant random thinning when interpreting complete-chain gains. Include
   whole-kernel costs; joint-query rates alone can exaggerate practical benefit.
   No force-screen model/protocol is implemented or frozen yet.
5. Existing `utility_onpolicy_v1` PHYSICAL proposals provide a separate internal
   recorded-proposal check of frozen screens without new oracle calls. Freeze
   that analysis before evaluating the models there and retain all 1,152 physical
   attempts. Report expected utility/cost, not actual savings or a screened-chain
   result. Never fit on these outcomes.
6. Keep every failure and prior negative result. No fitting on internal selection
   outcomes, fresh follow-up outcomes, the old six-composition evaluation, the
   48-parent scalar-chain cohort or the 722 reserved conditions. A useful method
   still needs real chain gains, complete timing/data costs, strong physical and
   representative learned-generator baselines and an independent final cohort.
   Generic delayed acceptance and force Taylor estimates are not new principles.

Training 46196124 and audit 46196212 are terminal. Re-query Slurm before action.
The tensor oracle's noisy-output buffering failure is reproduced and repaired;
energy RPC time is now recorded separately. Existing runs keep their original
records and cannot acquire retroactive timing. See STATUS for the latest PDF.
Prior NEXT: `notes/archive/next_through_action_geometry_complete_20260912.md`.

Verified development PDF: `runs/verification/delayed_screen_completed_20260912/main.pdf`
(9 main pages, 22 total). Build record:
`research/evidence/delayed_screen_completed_build_20260912.json`. This does not
establish scientific submission readiness.
