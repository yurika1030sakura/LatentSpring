Latest2026-09-19 result: trajectory_connection_v1 is COMPLETE and audited.
Force-head joint yield43.55% versus parent42.38%: +1.17pp [-1.17,3.32], positive
in both fits but NOT a confirmed parent gain. Versus the matched FM head the
improvement is+2.34pp [0.39,4.49]. Work adds no confirmed gain. Main-method
adoption remains false. Read research/TRAJECTORY_CONNECTION_STATE_20260919.json
and notes/trajectory_connection_20260919.md. All jobs are terminal; costs are
complete in registry v26. There is no BGFM experiment currently queued.
Before scaling this small head, examine its limited TRAIN fit (about5-6% error
reduction for force supervision). Do not repeat the same recipe or call the
positive control contrast a proven advantage over the parent or GAGA.
The manuscript remains the published build v7: GitHub PDF and43 Overleaf files
were verified; publication_sync_v7.json records the release. The new experimental
head has not replaced the supported paper method. Historical running statuses
below do not override this checkpoint.

Current2026-09-19 checkpoint: physical_connection_v1 is COMPLETE, with no parent
improvement (32.62% versus33.40% joint GFN2 force<=5 yield). Read
research/PHYSICAL_CONNECTION_STATE_20260919.json; all costs are in registry v24.
The next bounded candidate is trajectory_connection_v1: the same frozen-parent
head learns from actual TRAIN parent states, with matched force and complete
local-work targets. Protocol is frozen before generation. No thermal or useful
neural work advantage has been established. Publication build v7 adds the broad physical-target control appendix and is
pushed/verified on GitHub c1de73d and Overleaf9805067. See publication_sync_v7.json.
Trajectory teacher47257342 is COMPLETE:256 FIT outputs,512 states and9216
physical queries, fully replayed. Local particle ESS median1.37/8; no fallback
states. Costs are included in registry v25. Fit jobs47258410_0 and47258728_1
have completed all six2000-step fits and now generate raw validation outputs.
Source7835673 is frozen; automatic GFN2 and audit47258729 follow. Read
research/TRAJECTORY_CONNECTION_STATE_20260919.json before resuming.
Earlier running-study statements below are historical, not live status.

# Current next steps

- Read research/PHYSICAL_CONNECTION_STATE_20260919.json. Monitor47245883 and
  its audit47245885; preserve the frozen protocol and all controls.
- The preceding128-composition study is complete: physical labels beat matched
  reference-only FT, but parent superiority remains uncertain. Do not relaunch it.
- The correction head reuses the teacher bank; it makes no new teacher calls.
  It freezes parent tensors and learns a bounded equivariant pair residual from
  current/parent-predicted geometry. No supplied graph or oracle at inference.
- Inference accounting is128 backbone calls PLUS64 small-head calls for EVERY
  comparison model, with zero heads attached to the controls.
- New broad-target results have a scoped appendix section; build and publish it
  once the current paper revision is ready. The previous published figures stay.
- Preserve Jarzynski's local scope; the current empirical-reference bank is not
  a thermally calibrated multi-basin distribution.
