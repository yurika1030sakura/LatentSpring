# Active research status — September 8, 2026 (US Eastern)

**Not yet submission ready.** The corrected implementation is running, but no
molecular benefit of the corrected energy objective has been established.
The original scalar-readout claims cannot be promoted to likelihood or
Boltzmann sampling claims. The user has authorised framework reconstruction;
authors and submission accounts are outside this execution task.

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-research`. Frozen audit: `../audits/bgfm_20260908`.
Every SLURM job uses a committed source snapshot; `jobs.jsonl` records it.

## Completed evidence

| Check | Outcome | Evidence |
|---|---|---|
| Corrected full-state CNF and real FlowMol gradients | 87 tests pass, including conditional FM regression and actual network backward | `tests/`; `/tmp/bgfm_research_full_tests.log` |
| FM plus corrected energy optimizer smoke | 20 steps, 20 energy contributions, finite weights; 5.38 GiB reported peak | `runs/runtime_smoke_v3/`, job 45570868 |
| Equal trace-budget controls | Squared/product × independent/common probes all complete 20 steps, finite weights | `runs/estimator_smokes_v1/`, job 45572136 |
| Analytic Gaussian mechanism, five seeds | Mean exact KL: squared independent 0.7476, product 0.00138; exact, common-probe and rotated-squared controls reach zero | `runs/toy_trace_v1/toy.json`, job 45572134 |
| Old-checkpoint resolution panel, eight parents | Two parents have 64→128 mean centred log-density drift of 90.36 and 66.18 nats | `runs/numerical_panel_v1/panel.json`, job 45569978 |
| Composition census | Scan 3,902,107 training structures; 6,527 validation structures have absent training stoichiometries | `runs/composition_audit_v2/composition_split.json`, job 45572131 |
| New conditional position FM pilot | 1,000 optimizer steps, all weights finite, 104.2 s training, peak 5.06 GiB | `runs/position_cfm_v1/validation.json`, job 45575822 |

The estimator controls have matching initial weights, FM random streams and
trace budgets. The independent-probe arms each skip one energy outlier; the
common-probe arms skip none. Their raw means are **not** comparative molecular
performance. The analytic common-probe control is as strong as exact tracing
in this affine example and must not be omitted. Independent products remove
stochastic squared-loss bias on a fixed trajectory, not ODE integration bias.
The Gaussian example is constructed and does not predict a molecular effect.

The eight-parent panel uses probes fixed across time and solver resolutions.
Training resamples probes over steps. Therefore its estimated noise magnitude
cannot be transferred directly to training. At 128 steps, the plug-in
squared-residual noise fraction spans roughly 4–54% across this small panel;
the two unresolved integrations preclude a calibrated-density interpretation.

The composition-disjoint development pool contains 59 structures with at most
12 atoms, 304 with at most 24, and 6,527 with at most 200. Existing perturbation
shards contain respectively 14, 70 and 1,610 matching absent-composition
parents. These are development cases drawn from the old validation set, not a
new blind test set. The old test file duplicates the validation file exactly.

## Current experiments and architectural change

The geometry branch has the explicit factorisation `pi_frozen(m) q_theta,T(x|m)`.
A frozen original checkpoint supplies a possible composition prior. A separate
fine-tuned checkpoint supplies the memoryless, unaligned position flow.
Position FM ends the **data** at T=0.8 using the correctly rescaled head target;
it does not merely stop the original time-one interpolant early. See
`notes/factorized_geometry_flow.md` and `cfm_mol/clamped_fm.py`.

- Job 45575823: float64 adaptive trajectory plus independent divergence
  quadrature, easy parent and two worst old-checkpoint cases. Results persist
  after each solve, including budget failures. `runs/adaptive_panel_v2/`.
- Job 45576075: matched Gaussian priors, 64/128-step conditional samples,
  eight randomly selected composition-disjoint parents, four samples each,
  old checkpoint versus 1,000-step position pilot. `runs/position_samples_v1/`.
- Prepared independent GFN2-xTB evaluation retains every failure and declares
  minimum electron-parity spin, because original spin metadata is unavailable.
  This evaluates strain under an explicit computational convention, not the
  missing original DFT state or global Boltzmann populations.

## Failed attempts retained

- 45568618: progress-bar callback contradicted disabled progress bar; fixed.
- 45569977: OOM from diagnostic force graphs at zero force weight and dynamic
  edge budget overriding apparent batch size; fixed before successful smoke.
- 45570869: torch 2.2 mmap requires a string path; fixed before full census.
- 45574453: adaptive reference completed the first rtol=1e-5 solve (2,078
  position evaluations), then exceeded 5,000 evaluations at rtol=1e-7. The
  old script had not persisted that first estimate; no density values can be
  recovered from its success log alone. The corrected runner saves each result.

## Gates before larger energy training and paper claims

1. Coordinate and density convergence for the new conditional baseline.
2. Useful generated structures and independent potential evaluations, including
   failures, paired common-prior comparisons and composition-level reporting.
3. Matched FM-only, squared, product, common-probe, shuffled-label and zero-label
   controls; multiple independently trained seeds; identical sample budgets.
4. Preserved charge/spin/source identity and fresh test data. Raw OMol directories
   are empty and the official Hugging Face dataset requires authorised access.
   A request for machine authentication or an existing raw-data backup is pending
   with the user. No credentials are requested in chat.
5. Baseline coverage against current energy-guided flow work, not only archived
   in-house variants. LDR already supplies the off-policy log-dispersion idea;
   the replica identity is standard and alone is not a contribution.

Official deadlines are September 18 (abstract) and September 25 (paper),
23:59 AoE: https://iclr.cc/Conferences/2027/AuthorGuidelines . See `PLAN.md`
for the internal evidence decision and drafting schedule. A passing code test,
toy result or formatting check does not satisfy the scientific gates above.
