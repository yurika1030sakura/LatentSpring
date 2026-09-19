Current2026-09-19 checkpoint: physical_connection_v1 is COMPLETE, with no parent
improvement (32.62% versus33.40% joint GFN2 force<=5 yield). Read
research/PHYSICAL_CONNECTION_STATE_20260919.json; all costs are in registry v24.
The next bounded candidate is trajectory_connection_v1: the same frozen-parent
head learns from actual TRAIN parent states, with matched force and complete
local-work targets. Protocol is frozen before generation. No thermal or useful
neural work advantage has been established. Publication build v7 adds the broad physical-target control appendix and is
pushed/verified on GitHub c1de73d and Overleaf9805067. See publication_sync_v7.json.
Trajectory teacher47257342 is running; a two-seed array submission hit the
gpu_test per-user submit limit and must be retried after teacher completion.
The frozen fit source is runs/trajectory_connection_v1/fit_source (7835673).
Read research/TRAJECTORY_CONNECTION_STATE_20260919.json before resuming.
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
