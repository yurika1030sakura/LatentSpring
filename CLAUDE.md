# BGFM project guide

LATEST: tree-coordinate flow is COMPLETE and its chemical-validity gate FAILED.
Read `notes/tree_manifold_results_20260914.md`. Current candidate is the separately
frozen geometry/latent-relation feedback study; read
`research/GEOMETRY_FEEDBACK_STATE_20260914.json` and `notes/geometry_feedback_v1.md`
first. It restores a correctly parameterized self-conditioning baseline and tests
unlabelled pair-code learning through coordinate error. No improvement claim yet.


LATEST: dynamic-tree attention is COMPLETE with a failed gate; see
`notes/dynamic_tree_results_20260914.md`. Next active candidate is the coordination
prior plus geometry-preserving tree-coordinate flow. Read
`research/TREE_MANIFOLD_STATE_20260914.json` and `notes/tree_manifold_brief_v1.md`
first. It has a tested implementation and frozen matched controls, no improvement
result yet. Earlier source-utility and dynamic-attention failures remain unchanged.


CURRENT UPDATE: the user asked to prioritize a concrete new AI method and fast
improvement after the failed source-utility result. A state-dependent global-tree
attention block is now implemented. Read `research/DYNAMIC_TREE_STATE_20260914.json`
and `notes/dynamic_tree_attention_v1.md` first. No improvement is claimed yet.
The previous source-utility failure and all data protections below remain in force.
The new architecture changes the decoder; its law has no source-utility KL guarantee.


CURRENT: read `research/SOURCE_UTILITY_STATE_20260914.json`, `research/NEXT.md`,
`CLAUDE_HANDOFF.md` and `notes/source_utility_results_20260914.md`.

The frozen-decoder source utility method is implemented and the bounded pilot is
COMPLETE, with a failed development gate. It changes only latent-tree affinities,
uses exact source marginal importance ratios and a0.25-nat per-condition mixture
trust bound. Generic source adaptation, importance/KL identities and tree priors
have prior art, including ICLR2026 SGFM and Noise PPO. Originality/usefulness are
not established merely by this implementation.

Twenty-four new FIT and12 source-head held-out compositions were selected from the
verified real TRAINING corpus, before generation and disjoint from previous
candidate/panel compositions. They are NOT globally model-pretraining held out.
All36 reference assays and9216 source/structural attempts replay. The6144 new
validation outputs stay outside fitting; only3072 FIT-bank outputs are fit data.
All frozen decoder state tensors remain exactly unchanged.

Graph counts/768 for fixed/actual/shuffled/NLL:461/462/423/476 and410/392/410/409.
Pooled actual-minus-fixed is-1.11 pp, paired95[-3.84,1.76]; actual-minus-shuffled
is+1.37 pp[-1.63,4.36]. Both descriptive composition intervals also span zero.
Actual training-bank utility improves, but independent benefit does not repeat.
No new molecular oracle calls. Jobs46434075_0/1 completed0:0 in24:02/24:43.
Twelve source-utility/tree tests pass; the source heads have1881 parameters.

Preserve the completed monomer benchmark and all prior negative results in
`research/MONOMER_STATE_20260914.json` and the earlier notes. The earlier21 monomer
compositions are disjoint from two known processed corpora; do not confuse them
with this pilot's12 training-corpus-derived head-held-out compositions. Static
context and unchanged coordinate-NLL scale-ups remain closed. Do not grow this
utility recipe, change the reward or relax trust based on its validation outcomes.
A FIT-only check of reproducible source-affinity utility signal is the next bounded
question, before another method/training run; no new performance experiment is live.

The manuscript `paper/tree_working.tex` includes this failed pilot and checked prior
art. The ICLR goal remains unachieved and scientific submission readiness false.
Preserve722 reserved outcomes, old evaluated cohorts,2560 orbit outputs,36352
source/monomer evaluation outputs,32 older raw references,78 monomer-pool references
and the new12 held reference geometries outside FIT. Keep OMol25 primary,
max_atoms200, bond loss zero, original charge/spin, BGFM interfaces, separate
environments and no home writes. No final-density, output-ESS or Boltzmann claim.
The user handles submission. Query Slurm before assuming later work is absent.

Historical milestones below do not replace this checkpoint.

## Active takeover branch

Current review branch: `iclr2027-arch-fix` (Claude architecture update, then
Codex audit). Read `notes/claude_update_review_v1.md` before using the optional
index-split diagnostic or interpreting its particle results. Main molecular
production remains on the original immutable snapshot.

The original48-arm campaign and all8 geometry conditions are now complete.
The active reconstruction is described in `research/RECONSTRUCTION_CURRENT.md`:
conditional proposal densities, explicit validity support and reversible
chemical/coordinate moves. These are research prototypes. The uniform chemical
move has a useful molecular diagnostic; a learned-policy or ICLR-level advantage
has not been established. `research/NEXT.md` is the current execution checkpoint.

**Claude continuation entrypoint:** read `CLAUDE_HANDOFF.md`, `research/NEXT.md` and
`research/CHEMICAL_WORK_POLICY_STATE_20260913.json` first. They record the current method,
live job handles, exact next commands and remaining scientific gates. The core
candidate is ready for continuation; the paper is not submission ready.
Historical checkpoints below and in STATUS must not replace live scheduler
verification. `research/NEXT.md` is the current action list.

The user's latest priority correction takes precedence over older next-action
lists: focus on AI framework novelty and a fast real molecular demonstration.
Do not require all-case optimization or perfect mixing first. The19-arm optimizer
recovery is paused before submission. A new edit-conditioned reversible transport
is implemented and training46215808 is complete for two seeds and three model
ablations. Read `notes/edit_conditioned_bridge_v1.md` and the current NEXT.
Its usefulness and distinctive AI contribution remain unproven; test actual
corrected proposals next. Keep essential correctness, costs and data isolation.

The user's September12 scope clarification is authoritative for this campaign:
read `research/CLAIM_AND_BENCHMARK_SCOPE.md`. Include learned-generator baselines
as well as physical controls and evaluate realistic reuse costs. Universal
perfection, success on every element and an independent new physics law are not
submission requirements. Correctness and evidence for the actual claims remain
mandatory; a four-parent all-cost stress test is not the sole readiness criterion.

The current learned-guide manuscript is `paper/angular_working.tex`, with its
own angular/joint-geometry methods, proofs and completed pilot evidence. It is an
explicit development draft. Normalized learned proposals repeat short graph first
passages, but strong physical controls, preparation cost and poor distribution
diagnostics prevent a superiority claim. Pendant-fragment maps are implemented
and their physical pilot is audited, including a preserved Al validator failure.
The direction-augmented EACF MH baseline and its full real-map replay are complete.
These are frozen refinement checkpoints reused as proposals, not native EACF/FAB
performance. The original 72-parent concentration-10 benefit and its independent
GFN2 ordering remain recorded, but the completed concentration-64/400 controls
remove the demonstrated competitive advantage. All 28 strong-control arms, their
full replay and the cost-accounted summary are complete. Frozen six-composition
transfer also has no demonstrated average gain. The stiffness student overfits;
direction contributes over 98% of its held-out-parent local-teacher KL.
Eight disjoint training compositions have now been generated/audited, with two
zero-support cases retained. The constrained geodesic proposal now has fully
audited root and joint graph/radius/two-root implementations. On real training
starts, joint support increases from690/768 to745/768; all3,072 attempts replay
and750 independent forward/reverse densities agree. This is geometric progress,
not a demonstrated learned advantage. Local force/work fits fail as unrestricted
probability teachers. On valid arcs their higher acceptance still gives less
expected one-step energy decrease than uniform arcs; do not blindly distill them.
The complete physical-chain pilot46176878 and audit46176906 are complete:
48 FIT parents,36,864 calls,4,668 independently checked joint ratios. Site arcs
improve the128-call training potential-change readout by0.07615 eV over legacy
site64, with22.4% more sampling runtime. Uniform arcs do not establish average
full-chain gain. This remains a physical baseline result.

The new scalar conditional energy module is now implemented, trained and audited.
`cfm_mol/conditional_arc_energy.py` predicts masked-context radial interaction
curves; `arc_energy` integrates its actual scalar score into the full joint
proposal. Work-only and work-plus-force objectives each have two frozen800-step
models. All four improve internally withheld nonlocal work prediction by roughly
19--24% relative to the physical site score; this is not a molecular sampling gain.
The protected438-context data and all final model/baseline metrics are replayed.
The added928 withheld endpoints use1,856 raw calls; training uses no new oracle.
All jobs are terminal:46178737/46178918 and46179382/46180176. Read
`research/CONDITIONAL_CHAIN_RESULT_20260912.json`,
`notes/conditional_nonlocal_arc_learning_v1.md` and the updated NEXT before action.

The scalar learned-chain comparison is now COMPLETE on48 internally withheld
parents and six compositions. All36 arms reach their declared caps and8,682
joint MH ratios pass independent checks. Work and work-plus-force lose the
model-data-cost comparison by+0.5453 and+0.4929 eV, respectively; both intervals
exclude zero. At equal128-call inference, differences are+0.0469 and-0.0055 eV,
with intervals crossing zero. The force-loss addition has no established gain.
Do not scale these frozen models or call their19--24% work-prediction improvement
an effective sampler. Canonical-move diagnostics also show no clear benefit.

A near-pole cancellation was repaired without relaxing the frame check. The
failed learner reuses352 cached calls. One affected physical-control suffix had
to be regenerated for consistent arithmetic, adding1,392 calls. Corrected
trajectories use66,588 calls; actual research expenditure is67,980. Original
failures and discarded suffixes remain preserved. Partial-arm timing excludes
its previous failed-attempt time, which is separately recorded; no matched-wall-
time claim follows. See `research/CONDITIONAL_CHAIN_RESULT_20260912.json`.

Bounded geometric accepted-utility learning is now implemented, trained and
audited. It differentiates observed forward/reverse joint proposal densities
while retaining no-grad production draws. Two seeds improve the internal offline
utility-per-call proxy by about 2.7%, with stable importance weights. Its physical
behavior data costs 18,110 raw calls, not the scalar model's old overhead.

The fresh actual-oracle check is COMPLETE: 72 internal source states, 3,456
attempts, 6,708 raw calls and 3,354 independently replayed MH ratios. Actual
utility differences are near zero and both parent intervals span zero. Small
effects remain possible, but no fresh-proposal or full-chain benefit is
established. Do not scale this frozen geometry-only recipe. Those preceding geometric-utility jobs
are terminal; the new comparison below is active. Re-query Slurm. Read
`research/ACCEPTED_UTILITY_STATE_20260912.json`.

The six-arm bounded CONDITIONAL action/geometry comparison is now COMPLETE.
Neither action-only nor joint establishes an internal utility gain versus
physics. Geometry-only reproduces the prior +2.7% proxy, whose fresh-proposal
benefit remains unqualified. All final/baseline metrics and576 full densities
replay; eight trained-head gradients pass finite differences. Jobs46192301 and
46192339 are terminal. Read `research/ACTION_GEOMETRY_STATE_20260912.json`.
Do not scale these frozen weights or claim useful coupling from the architecture.

The bounded delayed-screen comparison is now COMPLETE: two linear and two
neural300-step models with fixed zero/physical controls. All metrics and6,388
supported-pair cases pass independent checks. Learned internal query-rate gains
are31/43% over no screening, but the fixed physical screen has a higher93% point
gain and neural-minus-linear intervals span zero. No useful neural contribution,
actual saved calls, complete-chain or wall-time gain follows. Jobs46196124 and
46196212 are terminal. Read `research/DELAYED_SCREEN_STATE_20260912.json`.

Actual dispatch screens before the candidate oracle query and distinguishes
valid geometry from scored proposals. The zero screen reproduces previous RNG
streams, queries and states. A fake-worker test exposed buffered stdout hiding
an oracle response; the byte-buffer reader fixes it, and RPC wall time is now
recorded separately. Original physical results and old timing are retained.

The general source-force screen and whole-prefix accounting comparison are now
COMPLETE. Forward decisions use only cached source force; candidate force is
used after querying for the reverse gate. The correction includes both actual
gate probabilities. Four300-step linear/neural models and fixed physical/work/
zero controls plus FIT-matched thinning are audited. All6,388 pair cases pass
independent checks, and all whole-prefix metrics replay. Both neural seeds lose
to the fixed physical screen; only one beats thinning. No repeatable useful
neural advantage is established. Do not scale these weights. Jobs46199581 and
46199747 are terminal. Read `research/SOURCE_FORCE_SCREEN_STATE_20260912.json`.

The fixed-source whole-prefix proxy includes initial/nonjoint costs but does
not replay changed chains. Neural gains6.06/2.19% versus no screen are below the
fixed physical13.57%. The earlier93% value used conditional joint-query rates;
it is not a whole-sampler speedup. No new physical queries or actual savings
occurred. Force provenance and original36/12 split remain audited.

The matched800-step direct versus gate-pretraining comparison is now COMPLETE.
All eight models and boundary/final objectives are audited, with matched sampled
index streams. Pretraining does not resolve the original12-parent selection
failure. The larger48-parent, six-composition reused evaluation is also complete:
no screen establishes a benefit over no screening. All30,058 model/pair cases
pass independent checks. The two unseen composition identities do not establish
consistent neural benefit. No new physical queries or actual savings occurred.
Read `research/GATE_DISTILLATION_RESULT_20260913.json`.

Stop scaling/tuning these screening-only recipes. Small positive point changes
against another screen do not replace the no-screen comparison. All current
jobs46203103,46203225,46206100 and46206282 are terminal; re-query Slurm before
recovery. The larger cohort is evaluation-only and previously served scalar
experiments; it is not an untouched final benchmark and must not enter fitting.

The128-cap mobility continuation and boundary checks are COMPLETE. All53 capped
arms were continued with cached-prefix replay, adding3,768 raw calls. Root and
collective convergence are63/64 and46/64;20/32 full pairs converge. The paired gap
contrast remains inconclusive. All14 minimum-step stops were checked with frozen
raw/distance-adjusted force steps, adding94 raw calls. Adjusted directions give
12 resolved decreases and2 increases. This motivates actual-energy backtracking
and constraint-aware optimizer recovery, not a new sampler or AI novelty. Read
`research/MOBILITY_CONTINUATION_RESULT_20260913.json` and
`notes/mobility_relaxation_continuation_decision_v1.md`. All four new jobs are
terminal. The former next recovery is paused by the user priority correction above.
Preserve the128-arm record, target, graph and electronic state. Keep evaluated cohorts and722 reserved outcomes out of fitting. The full
ICLR objective remains active and scientifically unachieved.
The previous manuscript is `paper/main.tex` with
`sections/M1_refinement.tex` and `M2_refinement_proofs.tex`, describing the actual
exact-entropy refinement candidate. It is explicitly a development draft.
`paper/legacy_audit.tex` preserves the historical audit entrypoint. The build
script accepts an optional second entrypoint argument. Do not confuse either
formatting pass with scientific submission readiness.

Read `research/STATUS.md` for completed jobs, failures, current experiments and
scientific gates; `research/PLAN.md` records the deadline plan. The user has
explicitly authorised framework reconstruction. The new position-only branch
in `cfm_mol/clamped_fm.py` and `scripts/research/train_clamped_position.py`
factorises a frozen composition prior from an unaligned conditional geometry
flow. Endpoint-head experiments end at T=0.8; a separately trained residual
velocity head ends at T=1 and records `position_parameterization=displacement`
in its checkpoint. Never guess the head semantics or silently reinterpret old
weights. See `notes/factorized_geometry_flow.md` and
`notes/displacement_geometry_flow.md` for the respective targets. No molecular
energy-training advantage has yet been established.

## Historical density-audit findings

The existing experiments show improved local ordering of an archived scalar
readout. They do not establish a calibrated density or improved generation.
The archived density routine integrates FlowMol3's endpoint prediction as a
velocity and uses a frozen-trajectory gradient. It remains available for
reproduction, with an explicit warning in energy training. Its historical
name `log_density_via_flow` is not evidence of likelihood correctness.

The new `cfm_mol/clamped_density.py` defines a separate deterministic,
zero-centroid clamped positional ODE, converts the endpoint head using the
actual schedule, and differentiates through the complete midpoint solve and
prior. Its default terminal time is 0.95. This is **q_0.95**, not the joint CTMC
sampler's conditional or endpoint density. No archived molecular performance
number came from this corrected path. It needs new training and convergence
validation. `configs/audit/omol25_clamped_cnf_smoke.yaml` is a smoke template,
not a production recipe or a reported experiment.

The later endpoint-entropy prototype in `cfm_mol/endpoint_entropy.py` uses an
actual-terminal-noise proposal score, with VSD/DMD and denoising prior art. It
is a gradient prototype, not a completed molecular method. The invariant
proposal-score critic failed independent qualification and both longer-training
seeds; no molecular actor was updated. The global innovation Gaussian auxiliary
also failed its bounded screen. Do not repeat these recipes or promote their
small-panel passes. Read `research/NEXT.md` and `research/NOVELTY_DECISION.md`.

The newer `cfm_mol/stein_calibration.py` adds conservative residual score
calibration. Its fixed four-feature version reduces estimated score risk on an
independent16384-parent panel but fails unfitted radial/angular checks. Only the
frozen calibration component has been tested; it does not qualify an actor
update or establish novelty. See `notes/stein_calibrated_entropy_candidate.md`.

The current exact-entropy refinement route freezes the FlowMol FM base and
learns a separate invertible adapter. The nonlinear species-coupling model,
invariant conditioner, exact centered volume and same-context affine/typed
controls are implemented. The actual source has cross-product features and is
not guaranteed O(3)-invariant. Current production uses an inversion-mixture
source and inversion-averaged potential; read the frozen parity protocol before
training or evaluating. The force average has a MINUS sign on the inverted force.
The quadratic target coefficient is 0.05, implemented as restraint/2 with
restraint_eV_A2=0.1. The one-query raw-energy training estimator is valid for a
linear expectation, not inside importance weights or MH acceptance.

The projected N8 re-score retains small convex-minus-affine changes of
-.013366+/-.002517 and -.012610+/-.002150 nat on shared2048 parents. Path ESS
remains approximately1/2048; no calibrated sampler or established ICLR novelty
follows. Keep the base frozen for unknown-source-entropy cancellation. The affine
control's full conditioned map remains nonlinear. Full eight-condition source
generation and the generic trainer are complete; corrected48-arm production
is submitted as45914819/45914826. These new FM64-plus-noise sources have no
evaluated path weights. Read NEXT and the handoff for current qualification,
external baseline environment status and remaining experiments.

## Locked project decisions

- Flow matching, bond-free: `total_loss_weights.e = 0` on OMol25.
- OMol25 primary data, `max_atoms = 200`.
- Loss family: FM + lambda_1 force + lambda_2 grouped energy (+ lambda_3 anchor).
- Reuse the installed FlowMol3 backbone. The existing joint pipeline remains
  patch-based; the user-authorised conditional branch may replace its training
  objective without modifying the shared external installation.
- Keep `cfm_mol/projection.py` and inactive bond-dependent code for later work.
- Preserve non-finite-loss guards. Do not infer achieved Boltzmann consistency
  merely from the loss family, infinite capacity, or the potential's coverage.

## Architecture and entry point

`scripts/run_train.py` reads config, **pops `mol_fm.bgfm` before
`model_from_config`**, constructs FlowMol3, calls `patch_flowmol`, then calls
`patch_flowmol_bgfm`. FlowMol3 rejects unknown constructor keys.

`cfm_mol/flow_model.py` patches interpolation, the inner endpoint-to-velocity
helper, sampling steps and optional discrete projections. Bond-free mode
disables discrete projections. The sampling retractions do not define a
smooth invertible CNF and are excluded from the corrected clamped density.

`cfm_mol/bgfm_train_hook.py` adds physics losses; perturbation shards are loaded
outside Lightning's DataLoader. Energy-only training requires those shards,
not force labels on the main FM batch. Historical force diagnostics remain
available; `force_endpoint_to_velocity: true` fixes head conversion only,
not finite-time force-target, aligned-prior or retraction assumptions.

`energy_density_options` selects the energy implementation:

```yaml
energy_density_options:
  mode: clamped_cnf
  terminal_time: 0.95
  parameterization: endpoint
```

Absent this block, legacy mode reproduces existing configurations and warns.
Changing modes changes the objective and requires separately identified runs.

## Theory references

- `notes/bgfm_method.md`: corrected score/density/grouped-loss specification.
- `notes/group_overlap_bound.md`: finite-set spectral-gap bound; standard
  mathematics, not a claimed new theorem or a continuous basin-mass guarantee.
- `paper/sections/A1_proofs.tex`: ideal residual results with audit caveats.
- `notes/archive/`: original April derivation and Appendix A v4 recovered from
  laboratory storage. They contain superseded claims and are historical only.

For an independent Gaussian linear interpolant the score readout is
`(t*v-x)/((1-t)*sigma^2)`, with velocity v, t<1, norm cap 1000.
The identity is not generally the score of an arbitrary learned ODE, and the
aligned/retracted production path needs separate justification.

## Environments and storage

The work-correction development branch is specified in
`notes/nonequilibrium_framework.md`: `cfm_mol/nonequilibrium.py` implements
existing finite-path importance weights and an AIS teacher, while
`cfm_mol/clamped_work.py` connects frozen conditional FM velocities to
Gaussian proposals in an orthonormal COM-free basis. This changes the sampler;
it does not retrospectively correct archived samples. Explicit spin, target
confinement and ESS diagnostics are required. No molecular advantage is yet
established, and AIS/SNF/FEAT identities are not claimed as new.

The eSEN checkpoint is available at
`/n/holylabs/woo_lab/Lab/yulili/bgfm/checkpoints/omol25/esen_sm_conserving_all.pt`
and has passed a CPU H2 energy/force smoke in `envs/omol25`. See
`research/evidence/oracle_availability.json`. The previous missing-checkpoint
statement was caused by checking an obsolete storage path. The public OMol training archive is now available under
`/n/holylabs/woo_lab/Lab/yulili/bgfm/raw_data/omol25/v250514/`. Exact replay has
restored metadata for every legacy record, with an integrity-checked read-only
index and original energy/charge/spin sidecars. Official validation recovery and
overlap auditing are also complete, with 664 development and 722 reserved
conditions. Reserved method outcomes remain unqueried; explicit source links
do not exhaust every parent-trajectory relationship. Read STATUS for full scope.

Do not merge the environments: flowmol (torch 2.2 + DGL + Lightning) is for
training/inference; omol25 (torch 2.8 + fairchem) is for preprocessing/oracle
queries. Never install fairchem into flowmol.

The active checkout is `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`.
Its new runs are stored locally under that laboratory path. Its environments
and processed-data links point into
`/n/holylabs/woo_lab/Lab/yulili/bgfm/`. The older ryl_lab tree contains the
April code and notes; do not assume it is the latest training checkout.
The repository's storage restriction prohibits writing into home. Stage
changes and builds in /tmp unless the user explicitly allows small code
edits in the current checkout; persist large outputs in laboratory storage.
Use `PYTHONNOUSERSITE=1` so a user-installed NumPy 2 does not break torch 2.2
and wandb. Set `PYTHONDONTWRITEBYTECODE=1` and a /tmp `MPLCONFIGDIR` for audits.

## Reproduction commands

From the intended checkout, with `FLOWMOL_PY` pointing to envs/flowmol/bin/python
and `PYTHONPATH` including that checkout (the environment has editable installs):

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/bgfm-mpl \
  "$FLOWMOL_PY" -m pytest tests/ -q -o cache_dir=/tmp/bgfm-pytest
python scripts/audit_iclr_evidence.py --out /tmp/bgfm-evidence
bash paper/build.sh /tmp/bgfm-paper-build
```

The evidence script needs only Python's standard library and checked-in files.
It verifies the original scalar while joining re-score records to energies,
uses checkpoint intersections across resolutions, drops reference geometries,
and applies the archived 93-parent mask. That mask is not an independent
molecular-identity split audit. It reports every seed and relaxation failure.
The build script exits nonzero for errors, unresolved citations, or >9 main
pages; passing this formatting gate does not certify scientific readiness.

## Data and runtime details

Read OMol25 labels with `atoms.get_potential_energy()` and `atoms.get_forces()`;
they live in `atoms.calc.results`. `n_atoms_histogram.pt` is `(values, counts)`.
Keep `python -u` and `PYTHONUNBUFFERED=1` in SLURM launchers. The valency JSON
is required by SampleAnalyzer, whose bond-free RDKit failures are guarded.
Use only within-parent energy variance; cross-batch offsets are confounded.
Charge clipping and missing spin in the historical preprocessing restrict
claims; a new chemistry-wide study needs preserved metadata and identity splits.
