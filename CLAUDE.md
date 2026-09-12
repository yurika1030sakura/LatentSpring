# BGFM project guide

Updated 2026-09-12 after constrained geodesic and complete joint-kernel audits. Read
`audit/20260908/REVIEW.md` before interpreting any result as Boltzmann sampling.

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
`research/GEODESIC_RECONSTRUCTION_STATE_20260912.json` first. They record the current method,
live job handles, exact next commands and remaining scientific gates. The core
candidate is ready for continuation; the paper is not submission ready.
Historical checkpoints below and in STATUS must not replace live scheduler
verification. `research/NEXT.md` is the current action list.

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
No new model is fitted. The complete-chain TRAINING pilot46176878 and audit46176906
are complete:48 FIT parents,36,864 calls,4,668 independently checked joint ratios.
Site arcs improve the128-call training potential-change readout by0.07615 eV over
legacy site64, with22.4% more sampling runtime. Uniform arcs do not establish
average full-chain gain. This is a promising physical baseline, not an AI result.
Start with `research/GEODESIC_RECONSTRUCTION_STATE_20260912.json` and
`notes/support_constrained_geodesic_proposal.md`. The prior conditional-learning
plan is partly superseded. The pinned connected validator has empty support for
training condition6 by a necessary degree bound; preserve that failure and the
other unresolved source zero. Do not scale the old learner or claim a successful
AI repair. Equilibrium, competitive learned-generator and matched-wall-time
superiority remain unqualified.
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
