# Response to the supplied manuscript review

The review at `review/review_20260915.md` is preserved verbatim from the user's
home-folder input. Its scientific priorities now govern this follow-up.

## Changes made to the manuscript

- The main story centers the composition-only latent-tree source and the
  physical-versus-replay update direction. The same-norm direct-update table is
  now in the main text.
- HarmonicFlow/FlowSite is cited from the ICML2024 PMLR record:
  https://proceedings.mlr.press/v235/stark24a.html. The distinction is an
  unavailable chemical input graph, not first use of harmonic/self-conditioned FM.
- The source-density equation is explicitly an analytic property. Primary FM uses
  source samples and does not evaluate the determinant; the physical teacher uses
  its separate Gaussian anchor reference q_A.
- The rotor58.5% result is removed from the abstract. The complete work/rotor
  development remains in the appendices; curvature ESS and its lack of established
  extra student benefit remain reported. No more rotor/curvature studies are planned.
- Absolute physical quality is visible in a new main table and appendix curves.
  This reuses the already audited raw24-composition outputs with zero new samples
  or oracle calls. GFN2 joint graph/force<=5 yield is35.68% for Gaussian,
  44.53% for harmonic, and47.59% for harmonic plus paired physics. The last
  increment is3.06pp, with composition95 interval[1.04,5.21] and positive changes
  in both continuations. Medians use each method's own valid subset; yields use
  all attempts. These are descriptive readouts, not an equilibrium test.

## Highest-priority experiment: complete the source/physics factorial

`source_physical_factorial_s{0,1}_v1.json` freezes the new Gaussian physical cell.
Each Gaussian base generates its own256 FIT starts on the same eight original
TRAIN compositions. The original eight-particle teacher recipe, step sizes,
training schedules and maximum preparation budget are preserved. Actual teacher
support and query counts are reported; starts are never replenished. Two1000-step
students are differenced with coefficient1, exactly as for the harmonic model.
The three old cells are reused. The new raw coordinates receive eSEN and
independent GFN2 readouts. This is a fixed follow-up on the previously evaluated
24-composition panel, not a newly untouched panel.

The initial gpu_requeue submission46643140 remained pending. To prioritize this
short study, the first GAGA seed was paused after its distance checkpoint reached
15000 steps; its nascent tree prefix is retained in the execution records.
Job46645569 runs this factorial first and then resumes GAGA seed0 from its saved
state. The other seed46635650_1 continues unchanged. The confirmation dispatch
46636422 now depends on these live jobs. No model/protocol choice changed.

The factorial auditor replays teacher proposals/support, uniform target selection,
paired parameter arithmetic, raw graph assays, parity-averaged eSEN values and
GFN2 output parsing. It reports within-source physical gains and between-source
gains with/without physics. A positive statistical interaction is not assumed.

## Replication and budget dependence

Array46643602 adds three prespecified primary-backbone continuations for both
Gaussian and harmonic sources. Along with the original two, there will be five
stochastic continuations of a shared pretrained model on two fixed3000-row blocks.
These are not five independent pretraining runs. Checkpoints at1000,3000,6000
updates are retained; all outcomes are reported and no checkpoint is selected.
At3000 steps each model generates32 outputs for each of the original64 test
compositions. The other budgets use16. This measures training-budget dependence,
not isolated dataset-size efficiency. Results/checkpoints are stored in netscratch
through `runs/source_replication_v1/results`.

## Remaining evidence boundary

The complete source-by-physics result and the extra repetitions are still pending.
The running matched EGNN/GAGA feedback study provides a stronger architectural
comparison, but does not yet supply a comparison where both final methods receive
matched physical feedback. That point remains open and must not be labeled solved
by the current main-backbone factorial or by the new absolute-quality summary.
Published result claims remain based on completed audits until these jobs finish.
