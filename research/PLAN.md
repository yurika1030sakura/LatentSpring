# ICLR 2027 research programme

Owner: Codex, taking over at the user's request on 2026-09-08. The user has
authorised framework changes and execution; authors/submission accounts are
handled by the user. This document records plans, not achieved results.

Deadline: abstract September 18; paper September 25, 2026, 23:59 AoE.
Source: https://iclr.cc/Conferences/2027/AuthorGuidelines (checked September 8).
Target an internal evidence decision by September 15 and a full draft by
September 20. Submit only claims supported by completed experiments.

## Scientific hypothesis to test

September 9 reconstruction: the user explicitly authorizes replacing the
theory, objective and framework. The original grouped-dispersion study below
is retained as an audited baseline, not an obligatory final paper narrative.
The current primary test is whether a properly normalized, symmetry-aware
proposal derived from a frozen FM pilot can support useful tempered correction
and subsequent FM distillation across molecular conditions without neural-flow
likelihood integration. AIS/SMC, matrix-Fisher identities, symmetry averaging
and weighted CFM are established tools, not sufficient novelty by themselves.

Current gates are global coverage/normalizer agreement, surviving diversity,
accurate electronic-state conditioning and matched oracle/compute performance.
An endpoint ESS increase after resampling does not pass these gates. The public
4M raw training archive has now been recovered; exact replay is restoring
source identities, unclipped charge, spin and original float64 energies.
Final evaluation must use a frozen, audited split rather than the duplicated
legacy test file. See `notes/tempered_sampling_protocol.md` and
`notes/rotation_mixture_protocol.md` for concrete protocols.

Can physically labelled off-policy molecular neighbourhoods improve an
amortised molecular flow when its position density is defined correctly and
both numerical integration and stochastic-trace errors are controlled?

LDR (arXiv:2602.03729) already covers off-policy log-dispersion regularisation.
The variance loss itself is not our novelty. Investigate the particular
failure of squared stochastic log densities: their expected loss includes a
parameter-dependent trace-noise penalty. Independent replica products remove
this penalty for a fixed deterministic discretised trajectory. This is a
standard cross-product identity; any contribution must come from its effect,
efficient implementation, principled validation and molecular usefulness.

Keep OMol25 primary and training bond-free, with max_atoms=200. A small-molecule
diagnostic cap is a computational subset, not a changed general model limit.
Use continuous clamped q_T only for its stated conditional task; assess the
joint de novo generator through independent generated-structure endpoints.

## Sequence and stop conditions

1. Runtime: complete real FM plus corrected-energy optimizer steps on a GPU,
   save finite weights and inspect actual energy contributions and memory.
2. Numerical: matched geometry/probes at 16/32/64/128 steps; exact traces on
   small examples; multiple trace repeats; centre within parent. Separate
   deterministic integration error from stochastic estimation variance.
3. Data: preserve source IDs, charge/spin and coordinate fingerprints. Audit
   old splits by composition and geometry; these fingerprints do not replace
   chemical identity. Do not call an old pretrained checkpoint identity-held-out.
4. Estimator: exact analytic tests for noise bias and independent products;
   include ordinary squared residual with equal total trace budget. Optimizer
   stability is a required gate: cross-products can be negative and noisy.
5. Pilot: matched initial weights and batches; FM, value, scrambled and zero
   labels; independently repeated runs. Freeze tuning before final test.
6. Main: expand only if calibration or independent sampling benefits survive
   matched resolution and equal-compute controls. Use seeds as replication
   units; report all failures, convergence, size/composition and compute.

No automatic rule converts a successful smoke test, positive toy result,
or local rank correlation into an ICLR-ready molecular result. Record negative
evidence and change the research direction when the gates fail.

## Execution discipline

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`.
The September 8 audit delivery is frozen. Home checkout is not modified.
Every new SLURM job uses a committed immutable source snapshot and a separate
output directory. Existing unrelated jobs are not touched. Preserve separate
FlowMol and fairchem environments. Record submissions and outcomes in
`research/jobs.jsonl` and `research/STATUS.md`.
