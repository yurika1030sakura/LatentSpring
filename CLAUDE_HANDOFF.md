# Claude continuation — cooperative interaction learning

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Never write home or merge the FlowMol/oracle environments.
Read current NEXT, state and theory note before running anything.

CURRENT: read `research/EDIT_INTERACTION_STATE_20260913.json`, `research/NEXT.md`
and `notes/cooperative_interaction_work_v1.md`.

The cooperative-interaction prototype is implemented and fully tested. Four-state
FIT data reveal real electronic coupling (parent-balanced mean absolute0.16419 eV).
All12 interaction models train/replay; the internal diagnostic has18 fit/9 held
parents, with9 original parents having no pair kept separately. Full models fit27
parents. The3D radial-contrast representation learns coupling; environmental
conditioning has no demonstrated advantage over simpler models.

A fixed-root-block pilot is complete and retained: many matchings are equivalent
under same-element atom relabelling, and utility differences are negligible.
The retained-panel kernel now chooses across different root blocks, uses the
SAME panel in the reverse normalizer, and treats coupling as a symmetric edge
preference, NOT a directional total-work correction. All physical states and
4,400 MH ratios replay. This gives small2--4% point improvements within the
cooperative family, with uncertainty and no contextual-network superiority.
The strongest single-edit learned method still has higher mean one-step utility.
There is no overall generator/chain advantage or ICLR-readiness claim.

All current jobs are COMPLETE0:0:46296885/46297013 labels/audit,46297734 training,
46299227/46299395 fixed-block evaluation/audit,46299770/46299887 panel replay/audit.
This stage used1,770 new raw calls (934 labels+836 endpoint evaluation); panel
re-evaluation used zero. Keep722 reserved outcomes unqueried and evaluated parents
out of fitting. Optimizer recovery and old single-edit neural scale-up stay paused.
The full ICLR goal is active and scientifically unachieved.

The concrete next task is the bounded complete-chain comparison in NEXT. The
cooperative candidate is implemented and has a small within-family signal;
it has not surpassed the strong single-edit energy-descent result. Preserve both
facts. Do not invent a novel general principle from auxiliary MH, pairwise mixed
differences or composed moves. Do not refit evaluated parents or cherry-pick
neural comparisons over the stronger typed-radial control. Authors/submission
stay with the user. The user prioritizes core AI value and fast evidence, with
no universal-success or optimizer-cleanup prerequisite.
