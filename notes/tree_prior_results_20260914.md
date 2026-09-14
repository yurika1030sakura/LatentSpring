# Tree-prior evidence and research decision — September 14, 2026

The spatial-prior framework is implemented and its fragmentation mechanism has
repeated empirical support. Chemical graph acceptance does not improve reliably
across the two connected-data training continuations. The learned node/pair
affinities have not established an increment over the fixed spatial prior.
This is a useful research checkpoint, not ICLR submission readiness.

## Method and scope

The network learns latent connection preferences from species, composition,
charge and spin, fitting an exactly normalized spatial tree-mixture density to
training coordinates. No supplied chemical bonds enter training. The final
generator remains flow matching with typed assignment and shared Haar symmetry.
The tree is a dependence structure, not a chemical graph. The complete derivation,
implementation boundaries and primary prior-art links are in
`notes/tree_mixture_prior_v1.md`.

All continuations start from checkpoint92269b59, use3,000 real OMol25 examples,
and have matched data, backbone and update budgets within each comparison.
Learned priors additionally fit1,000 of those same examples;128 separate train-split
examples validate prior likelihood. Evaluation supplies composition and original
charge/spin. Each method generates64 samples per condition on the same eight
development compositions. This is conditional 3D generation, not a demonstrated
joint generator of chemical composition and electronic state.

The unrestricted pilot's actual1,000 prior-training examples contain580
disconnected structures;79 of128 prior-validation examples are disconnected.
These are sample counts, not an estimate certified for all OMol25. A single
connected-output criterion is mismatched to that unrestricted target. The revised
comparison selects the first3,128 connected/nonoverlapping training structures
in a fixed permutation (7,130 examined), using the existing1.25 contact and0.6
overlap factors. Every arm receives the same selected targets. Selection uses no
generated output, bond perception or energy. The confirmation uses another
permutation (7,019 examined), another continuation seed and fresh generation
streams, retaining the same warm checkpoint and eight conditions. It is not a
second independent pretraining run or a new composition test set.

## Structural results

Graph check passed /512, with every attempted output retained:

| Training domain / run | Warm | Gaussian | Fixed tree | Node tree | Pair tree |
|---|---:|---:|---:|---:|---:|
| Unrestricted pilot |78|76|94|92|93|
| Connected target, first seed |66|105|131|121|111|
| Connected target, confirmation |—|91|93|—|—|

The fixed-tree minus Gaussian graph pass-rate difference is+5.08 percentage
points [1.17,8.79] in the first connected run, but+0.39 [-3.52,4.10] in the
confirmation. Do not present the first interval as a replicated validity gain.

The geometric mechanism is more consistent. Disconnected outputs fall from192
to162 in the first connected run and243 to194 in confirmation. Geometry-supported
counts rise from314 to340 and266 to312, respectively. Node learning gives352
geometry-supported outputs in the first run, but121 graph-supported versus131
for fixed. Pair learning gives337 and111. The legacy graph check has radical and
metal representation limitations; all exceptions and old decisions are preserved.
Graph perception is not an electronic-validity certificate.

Connected-target held prior NLL/DOF is1.59362 fixed,1.58443 node and1.57507 pair.
Better source likelihood has not established better final graph acceptance.
Absolute NLL across unrestricted and connected datasets is not a method comparison.

## Physical readout

All2,560 outputs from the first connected run were scored with the same frozen
eSEN checkpoint and original charge/spin. Evaluating x and -x uses5,120 raw
queries. Define E+=(E(x)+E(-x))/2 and U=E+ +0.05||x||². No generated geometry or
condition was selected by energy, and no new energy labels were used for fitting.

| Comparison | Mean E+ difference, eV | Conditional paired95% interval |
|---|---:|---:|
| Fixed minus Gaussian |−0.395|[−0.907,0.135]|
| Node minus Gaussian |−0.624|[−1.084,−0.144]|
| Pair minus Gaussian |+0.032|[−0.494,0.582]|
| Node minus fixed |−0.229|[−0.739,0.269]|

Node versus Gaussian is a positive exploratory energy signal. It is from one
training seed, does not establish a node advantage over fixed, and is one of
multiple comparisons. Per-condition tradeoffs remain: fixed and pair increase
mean energy on the Re-containing condition by3.457 and2.936 eV. Those cases are
retained. Method-specific valid subsets have different denominators; their
conditional means are descriptive. Neither low energy nor normalized source
density proves Boltzmann-distributed final outputs.

All intervals bootstrap paired draws within each fixed composition, conditional
on fitted models. They are marginal, without multiplicity adjustment, and do not
cover generalization to new compositions or uncertainty over training seeds.
The energy confirmation seed has not been queried.

## Audit, artifacts and decision

All6,144 initial sources and final structural readouts were replayed across the
three studies. The energy audit checks all5,120 stored energy/force rows, parity
averaging, restraint calculations, source coordinates, original condition labels,
source-linked support masks and hashes. It does not independently re-query the
oracle, rerun optimizers or regenerate final neural outputs. Implementation-stage
testing passed53 targeted tests, including density normalization, Jacobians,
sampling, gradients, symmetry and wrong-prior guards.

- `research/evidence/tree_prior_results_v1.json` and `.csv`: all studies and decisions.
- `research/evidence/tree_prior_confirmation_audit_v2.json`: strengthened confirmation audit.
- `research/evidence/tree_prior_energy_audit_v2.json`: strengthened energy provenance audit.
- `research/evidence/generator_reference_registry_v2.json`: all checkpoint identities and source laws.
- `research/evidence/tree_prior_scheduler_v1.json`: nine submissions, ten completed tasks; no active tree job at the recorded check.
- `research/figures/tree_prior_v1/tree_prior_structure_v1.pdf`: both connected-data runs.
- `research/figures/tree_prior_v1/tree_prior_energy_v1.pdf`: every energy condition and conditional intervals.

Earlier auditv1 files are retained; v2 adds checkpoint/source-mask provenance
checks without changing any outcomes. The summary/figures can be rebuilt with
`python scripts/research/summarize_tree_prior.py` and a writable MPLCONFIGDIR.

Retain the explicit source-law framework and fixed-tree reference. Do not scale
the current pair head or advertise either affinity network as a proved improvement.
The immediate scientific question is why a better source fit and less fragmentation
do not reliably improve chemically supported generation. Before attributing the
effect specifically to tree topology, a covariance/scale-matched Gaussian is a
relevant missing control. Any subsequent change to the learned source needs a
specific hypothesis and a frozen bounded test against both Gaussian and fixed
tree; current development outputs may inform diagnosis but must stay out of fit.
No replacement architecture or additional experiment is declared complete here.

The framework has more method content than an auxiliary-loss-only story. Its
originality and competitive usefulness still require evidence against prior
structured-prior methods and learned generators. A new physical law and perfect
performance on every case are not required. Honest support for the actual claim is.
