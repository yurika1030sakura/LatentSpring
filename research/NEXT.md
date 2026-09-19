# Current checkpoint —2026-09-19

Read research/CONTEXT_CONFIRMATION_STATE_20260919.json and
notes/gaga_status_20260919_zh.md. Job47277628 runs two seeds: cached-feedback
internal validation, a parameter-matched wide-head fit, then a fresh common-panel
comparison of parent/pair-long/pair-wide/context/GAGA128/GAGA651. Outcomes pending.
The full FlowMol and GAGA systems have different architectures, training data,
histories and costs; this comparison is not a shared-backbone ablation.

Capacity study47274322 is COMPLETE and audited. At20000 updates, the contextual
head predicts physical corrections with about15% lower internal validation MSE
than the original small head. Its37586 parameters motivate the new37540-parameter
wide-pair control. No external generation improvement follows from this fit metric.
The previous trajectory head gave+1.17pp [-1.17,3.32] versus its parent, so its
primary gate was false. Work added no established neural gain. Registry v27
includes completed capacity costs; current validation/confirmation is not yet in it.

Shared-EGNN GAGA superiority remains unestablished: graph15.63 vs13.77% with
CI crossing zero; calibrated GFN2 joint4.25 vs4.59%. Do not compare the FlowMol
43.55% from another panel to these numbers. User wants Chinese answers and
substantive improvements, not small control gains presented as a GAGA victory.

Published paper remains verified build v7 / publication_sync_v7.json, with43
Overleaf export files. New candidate heads are not the manuscript's main method.
Never write to home, never push the home origin remote, and keep the two Torch
environments separate. For figures use .agents/skills/scientific-figure-design/.
Historical statuses below do not override this checkpoint.

## Next actions

1. Monitor47277628 and its dependent audit recorded in CONTEXT_CONFIRMATION_STATE.
2. Verify all fresh comparisons, including both GAGA call budgets, both seeds and
   all attempted structures. Use raw coordinates and fixed-coordinate GFN2 only.
3. Count cached validation2048 outputs and fresh confirmation3072 separately.
   Wide-control training adds40000 updates. Existing capacity80000 is already v27.
4. Promote a candidate only after actual generation evidence; retain the same-size
   control and do not attribute a full-system result solely to the small head.
5. Publish completed code/results. Change the paper only for a supported contribution.
