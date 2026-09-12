# Bounded gate-target distillation candidate

The source-force comparison is complete. Both neural seeds lose to the fixed
physical screen on the declared whole-prefix proxy; their differences versus
that control have intervals below zero. One neural seed improves on no screen
and matched thinning, but the other does not. Do not scale the frozen weights
or call this established AI benefit.

A FIT-only diagnosis in `runs/gate_teacher_diagnostic_v1/results.json` shows two
different behaviors. Neural seed0 retains98.95% of recorded expected accepted
moves but its query cost is7.26 times an inadmissible oracle-informed teacher;
seed1 retains75.67% and costs5.71 times that teacher. The physical screen retains
51.20% and costs3.00 times the teacher. These are FIT diagnostics, not a revised
success metric or evidence that one sampler mixes better. The teacher uses the
candidate's unknown energy and cannot be deployed before querying it.

## Elementary oracle targets and retention relation

For complete finite MH log ratio R and floor exp(-B), define

    log g_f* = clamp(R, -B, 0),
    log g_r* = clamp(-R, -B, 0).

These gates retain the original accepted probability min(1,exp(R)). Each gate
is as small as possible under the floor while retaining that pair's original
accepted probability: for R>=0, the forward gate must be1 and the reverse at
least exp(-R); for R<=0 the converse holds. This is an elementary consequence
of min(g_f,exp(R)*g_r), not a new MH theorem or a claim of achievable zero-cost
energy prediction.

If delta is the larger positive log-gate underprediction relative to these two
targets, then alpha_screen / alpha_MH >= exp(-delta). The general floor gives
only the coarse bound exp(-B). Empirical average label error does not certify
uniform delta on unseen states or a global mixing rate. `cfm_mol/gate_teacher.py`
and `tests/test_gate_teacher.py` implement and check these relations. The true
oracle ratio must be fixed and already queried when constructing training labels.

## Next bounded comparison

The direct utility optimizer has sparse accepted-flow feedback. The same paid
pairs also supply dense gate targets on both orientations. Test whether such
supervision improves initialization, followed by the existing whole-prefix
cost/utility objective. This is a hypothesis, not a proven explanation of the
seed instability or a new contribution by itself.

Freeze a matched total-step comparison before optimization: direct utility for
800 steps versus500 steps of gate-target fitting followed by300 utility steps,
for both the five-coefficient linear and paired-state neural models, two seeds
per case. The exact loss weights and sampling law are not frozen yet. Keep the
same force/geometry data, floor, fixed controls and FIT-calibrated thinning.
Include the equally trained linear control, and report retained flow as a
separate diagnostic while keeping whole-prefix work/cost the declared primary
proxy. Never promote retained acceptance to a mixing certificate after outcomes.

Only36 FIT parents may be used for either phase. No internal-selection, existing
fresh-proposal, prior molecular-evaluation or722 reserved outcomes may enter
updates. No new physical queries are needed initially. Do not re-run the old
300-step models or expand capacity without this declared comparison. Any useful
internal result still requires actual chain/query/wall-time validation and strong
learned/physical baselines. Standard oracle labels, pretraining and this simple
bound alone do not establish the missing ICLR novelty.

## Actual screened-run estimator contract

For fully scored recorded reference pairs, expected accepted utility uses
`g_forward * alpha_2 * reward`. In a real screened run, the first gate has already
been sampled and rejected candidates have no new energy. Use zero for those
rows and `alpha_2 * reward` for queried rows, averaged over ALL attempted draws.
Multiplying the queried-row value by `g_forward` again would count the gate twice.
The sampler retains ordinary R in `log_acceptance_ratio`; read its explicit
`second_log_acceptance` for this conditional estimate and `scored` for paid cost.
Realized trajectory energy changes are a separate valid readout. This distinction
must be tested when wiring the future physical evaluation driver.

## Implemented matched comparison

The protocol is now frozen in
`research/evidence/gate_distillation_training_protocol_v1.json`. Both recipes
use identical activity-stratified index draws at every step, with the appropriate
exact empirical weighting for each objective. Direct utility and distillation
both reset Adam at500. Teacher loss is normalized squared log-gate error on
both orientations, with invalid attempts contributing zero in the full mean.
No selection outcomes enter either phase and no intermediate checkpoint is
chosen by validation. A500-step checkpoint is saved solely for objective audits.

The trainer hashes all sampled index arrays; the audit reconstructs them and
checks the boundary/final hashes. It also verifies both phase-boundary and final
objective gradients, all gates/metrics and fixed/thinning controls. Eight tests
pass. Training/evaluation code is `train_gate_distillation.py` and
`audit_gate_distillation.py` under `scripts/research/`. Read NEXT for live jobs.
No new physical queries are required; no scientific improvement is implied by
implementation checks.
