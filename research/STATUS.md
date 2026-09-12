# Current research status — September 12, 2026

Authoritative checkpoint: `research/CONDITIONAL_CHAIN_RESULT_20260912.json`.
The ICLR goal is active and scientifically unachieved. The latest scalar predictor
has no established molecular sampling advantage and should not be scaled.

The complete comparison covers48 internally withheld parents in six compositions,
three methods and two seeds. All36 arms reach their budgets. Independent audits
replay every trajectory and8,682 joint MH ratios. At the declared method/data cost
of21,006 calls per replica (plus6,088 common preparation calls), work-only minus
physical potential is+0.5453 eV, descriptive95% parent interval[0.3630,0.7590].
Work-plus-force is+0.4929 eV,[0.3099,0.7052]. All six composition point estimates
favor physical site arcs. These are internal development comparisons.

At equal128-call inference, the differences are+0.0469 eV,[-0.0577,0.1783], and
-0.0055 eV,[-0.1499,0.1471]. Neither establishes average energy benefit. The
force-versus-work difference is also uncertain. These findings are stronger than
merely failing to amortize training on a small cohort: the same-inference signal
is unqualified as well. The19--24% internal work-prediction improvement remains
true but does not constitute a useful sampler result.

A post-hoc constitutional-move diagnostic gives expected accepted changes per
parent at128 calls of0.995 (physics),0.927 (work),0.969 (work-plus-force), averaged
equally over compositions. Mean visited connectivities are1.927,1.917,2.016.
There is no clear exploration benefit. These counts do not measure independent
samples or mixing; same-connectivity moves may still alter conformations. The
separate five-candidate diagnostic improves energy ranking but mostly worsens
confidence KL; it is not a continuous conditional-KL estimate or a proven cause.

Summary: `runs/conditional_arc_chain_summary_v1/results.json`.
Figure: `runs/conditional_arc_chain_figure_v1/conditional_arc_controls.pdf`.
Move diagnostic: `runs/conditional_arc_chain_move_diagnostic_v1/results.json`.

## Numerical repair and provenance

Original worker46181560_0 failed on near-pole tangent cancellation; worker1
completed. The dependent audit46181692 was cancelled. Reorthogonalization fixes
the numerical residual without weakening checks. Recovery46182765 replays all47
cached requests/352 raw calls and preserves earlier states/transitions. Audit
46182808 passes five conditions but detects a previously rejected reverse-frame
error in one physical control. Only that control suffix is regenerated as46183584,
reusing its358-call prefix and adding1,392 new calls. Final condition5 audit
46183645 completes. All current work is terminal.

Corrected method trajectories contain66,588 calls; actual physical research
spending, including the discarded numerical-control suffix, is67,980. Both are
reported. The recovered partial arm's earlier failure time is retained separately,
so per-method timing does not support a matched-wall-time claim. No old evidence
is overwritten. The corrected condition5 is `runs/conditional_arc_control_repair_v1`;
use its audit with the other five completed audits from46182808.

All source zeros and prior negative controls remain. Training support counts are
45,32,106,99,0,19,0,96 out of256 each. Condition6 has empty connected support under
the pinned builder; condition4 remains unresolved. Original charge/spin and722
reserved outcomes are preserved. The old vector learner's failure against
concentration64/400 controls is not overturned.

The development manuscript includes the scalar prediction and complete-chain
negative results, recovery costs and limits. Current build:
`runs/verification/conditional_chain_completed_20260912/main.pdf` (eight main
pages,19 total; no unresolved references or overfull boxes; result pages visually
checked).
NEXT specifies a bounded accepted-utility investigation using FIT-parent data,
not another run of the failed frozen learner. Historical status:
`notes/archive/status_through_scalar_prediction_20260912.md`.
