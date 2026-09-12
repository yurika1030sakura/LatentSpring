# Current research status — September 12, 2026

Current checkpoint: `research/GEODESIC_RECONSTRUCTION_STATE_20260912.json`.
The ICLR goal is active. The method is not scientifically ready for submission.
Read `research/CLAIM_AND_BENCHMARK_SCOPE.md` for the user's intended standard.

The new constrained joint proposal passes independent correctness checks. The
old normalized-site learner has no established competitive advantage against the
completed concentration-64/400 physical controls. Its original concentration-10
benefit and GFN2 check remain valid only for those endpoints; frozen transfer
has no demonstrated average gain. Do not repeat or scale the old learner.

The reconstruction now restricts root directions to analytically computed
feasible circle arcs and evaluates the actual normalized proposal. Complete graph,
two-radius and two-root updates include both oriented-circle preimages, Cartesian
radius factors and the marginal density over both decoder orders. Empty partial
contexts and zero reverse density are failures, without resampling until success.

On 96 prepared TRAINING parents, all 3,072 joint proposal attempts were replayed.
The new proposals preserve the requested exchanged graph and positive reverse
density for 745/768 attempts, versus 690/768 for legacy site64. The uniform arc
method rescues 157 proposals by marginalizing decoder order. Independent density
quadrature checks 750 forward/reverse values with maximum log error 4.27e-14.
This is geometric support improvement, not learned efficiency or equilibration.

The 438-context force/work probe fits are locally accurate but can be bad global
probability teachers: in 47 negative-axial contexts only16/376 unrestricted teacher
draws were supported. Arc restriction repairs this to376/376; all14,016 real root
arc draws preserve the graph. However, the true-oracle FIT-context single-step
screen favors uniform arcs for expected energy progress: -0.1440 eV versus -0.0453
eV for the fitted score, despite the latter's higher acceptance (0.725 vs0.248).
Do not train merely to maximize acceptance or imitate that local teacher.

The original eight training-source support counts remain45,32,106,99,0,19,0,96
out of256 each. Both zeros stay in the denominator. Condition6 (C4H10OFCl) has
empty connected support under the pinned RDKit builder: maximum degree sum31 is
below the32 necessary for a connected17-atom graph. This is a target-definition
incompatibility, not proof that the composition is chemically impossible. The
other zero-support case remains unresolved. The validator and panels are unchanged.

| Completed work | Producer | Audit / summary |
|---|---|---|
| Strong physical controls | 46147217 | 46148255 / 46148506 |
| Eight disjoint training sources | 46149838 | 46149969 |
| 96 training preparations;11,910 calls | 46150875 | 46150903 |
| 438 conditional contexts;5,536 calls | 46152939 | 46153206 |
| Real root arc support | 46157830 | 46159984 |
| FIT root oracle screen;1,648 calls | 46160646 | 46160899 |
| Real complete joint support | 46174307 | 46176033 |

A bounded complete-chain TRAINING pilot is submitted: producer46176878,
dependent replay46176906, immutable sourcebde3590. It compares legacy site64,
uniform arc and site arc at128 raw calls per parent,48 fixed FIT parents,
two algorithm seeds,36,864 maximum new raw calls. Common local and force-rotation
moves are unchanged. Query stopping measures finite-cost output, not stationarity.
Re-query Slurm and consume all four conditions before interpreting outcomes.

The manuscript `paper/angular_working.tex` already reports the strong-control
negative results. Its latest reviewed PDF is
`runs/verification/sharp_controls_20260912/main.pdf` (eight main pages); the new
arc round is not yet incorporated. No new model is fitted. Representative learned
baselines, useful held-out performance and distributional claims remain open.
Earlier status is preserved in `notes/archive/status_through_sharp_training_20260912.md`.
