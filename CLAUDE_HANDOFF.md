# Claude handoff — BGFM / ICLR 2027

**September11 continuation checkpoint:** read `research/NEXT.md` first. Full
and compact EACF controls, complete checkpoint replays and independent geometry
comparisons are now finished. They do not establish our method's external
advantage; the full-model20x timing ratio is not supported as a general efficiency
claim after the compact comparison. Main production has39/48 audited arms;
Slurm45914826_5/6 remain live at the latest check. The current7-page development
draft includes these adverse results. Preserve the original dated handoff and
JSON snapshot below as history; they are not the current queue state.

**Post-handoff Claude update reviewed:** read `notes/claude_update_review_v1.md` and
`research/evidence/claude_update_review_v1.json` (repository-relative). Fixed-index
splitting is a labelled diagnostic, not a permutation-equivariant repair.
Three loader-failed experiments were recovered without retraining; the two
colliding LJ13 arrays were stopped with evidence preserved. Condition0 of the
original molecular campaign is complete; refresh all other job states.

Prepared at the user's explicit request on September 10, 2026. Start in:

```bash
cd /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027
```

Original handoff branch: `iclr2027-research`; current review branch: `iclr2027-arch-fix`. The home checkout `/n/home04/yulili/bgfm` is old.
Do not write there. `CLAUDE.md` remains the authoritative project guide. This
handoff supplies the current execution state and supersedes older dated status
paragraphs about queued jobs and unfinished engineering work. Re-query live
state before acting; a handoff snapshot is not a scheduler monitor.

## What you are taking over

The user authorizes continuing the research, experiments, framework changes and
manuscript preparation toward ICLR. Authors and submission accounts are outside
this handoff. Preserve existing evidence and finish the present experiment
before changing its protocol. Do not wait passively for GPU jobs: baseline
qualification and assessment orchestration can proceed independently.

**The core candidate is implemented and ready for continuation. The scientific
framework is not yet validated to ICLR standard.** There is a small two-stream,
one-condition signal; broad usefulness, strong external comparisons, distribution
coverage and a defensible novelty claim remain open. Do not tell the user that
only writing remains, promise acceptance, or declare the research goal complete.

Working story: **Refining Molecular Generators with Exact Entropy Changes**.
A frozen implicit molecular generator supplies configurations; a learned
invertible, symmetry-preserving adapter improves a specified energy objective
using its exact entropy change, without estimating the base density or score.
The possible AI contribution is the molecular neural transport architecture and
its demonstrated utility. Change of variables, relative KL, convex flows,
symmetry averaging and nonequilibrium work identities are established tools.
AFM/Jarzynski motivation alone is not AI novelty. No calibrated Boltzmann
sampler or unconditional chemically valid generation claim is established.

## Read in this order

1. `CLAUDE.md`, this file, `research/HANDOFF_STATE_20260910.json`.
2. `research/NEXT.md`, `research/PAPER_STORY_CURRENT.md`; STATUS is a dated
   historical ledger, so its older pending statements are not current truth.
3. `notes/parity_refinement_candidate.md` and the frozen
   `research/evidence/parity_training_protocol_v1.json`.
4. `notes/species_coupling_adapter_candidate.md`,
   `notes/species_coupling_execution_v1.md`, `notes/species_affine_ablation_v1.md`,
   `notes/eacf_comparison_contract.md`.
5. `paper/main.tex`, `paper/sections/M1_refinement.tex`,
   `paper/sections/M2_refinement_proofs.tex` and evidence cited there.

## Implemented method and non-negotiable mathematical contracts

- Source: frozen displacement FM64 plus 0.025-A terminal COM Gaussian noise,
  followed by an independent uniform inversion sign per original parent.
  Its retained 1-eV neural input is a pretrained feature, not physical temperature.
  The new broad source has no evaluated absolute density or importance weights.
- Target: labelled, unweighted center-of-mass subspace; fixed atom composition,
  charge and spin; `kT = 0.025851999786435 eV`.
  `U(x) = E_plus(x) + (restraint/2) sum_i ||x_i||^2`, with
  **`restraint_eV_A2 = 0.1`**, hence quadratic coefficient **0.05**.
  This is a confined ML-potential ensemble, not direct DFT or experiment.
- The frozen base is essential: unknown source entropy cancels when comparing
  `KL(T#q0_plus || pi_plus)` with `KL(q0_plus || pi_plus)`. A trainable base needs
  a new derivation. A finite sample estimate of this expectation is not absolute KL.
- Main files: `cfm_mol/species_coupling_adapter.py`, `centered_convex_flow.py`,
  `affine_species_adapter.py`, `linear_entropy_adapter.py`,
  `entropy_adapter_io.py`, `entropy_source.py`, `parity_refinement.py`.
- Element-group internal maps and centroid-pair maps use invariant neural
  contexts. For a centered active group with point-map Jacobians A_i, intrinsic
  log volume is `sum_i logdet(A_i) + logdet(mean_i A_i^-1)`.
  **Do not drop the second term.** Only 3-by-3 factorizations are needed.
  Inactive context must not secretly depend on active internal coordinates.
- The affine ablation keeps the same neural context, parameters and schedule,
  linearizing the active point map. Its full conditioned map is still nonlinear.
  The typed global linear adapter is a separate, weaker control.
- Public inverse is a no-grad reconstruction interface, not a differentiable
  inverse-likelihood API. Homogeneous/passive-poor contexts limit expressivity;
  no universal expressivity result is claimed.

### Symmetry repair: read before touching training or baselines

The actual FlowMol source uses `n_cp_feats=4` and cross-product features. It is
SO(3)-equivariant; do not infer O(3)-invariance of its distribution from our
adapter. An equivariant bijection cannot create a missing distribution symmetry.
Non-equivariance of a source point map alone also does not prove law asymmetry.

Use the explicit new protocol:

```
q0_plus = (q0 + inversion#q0) / 2
E_plus(x) = (E_raw(x) + E_raw(-x)) / 2
F_plus(x) = (F_raw(x) - F_raw(-x)) / 2
```

The force MINUS sign is required. Source mixing is stochastic, not a
zero-entropy-change bijection. Its separate KL benefit toward an even target
is JS(q0, inversion#q0), between 0 and log(2), and has not been estimated.
Transport gains reported against q0_plus are not the total gain from raw q0.
The current coordinate domain has no fixed absolute-stereoisomer restriction;
do not apply the mixture blindly to a chiral restricted domain.

Training may query raw energy once on each independently inverted parent:
linearity and invariance make the expected energy/gradient correct for E_plus.
**This shortcut does not work inside exponentials, FAB/AIS weights, HMC/MH
acceptance or projected-energy histograms.** Those require actual E_plus or a
separately justified estimator. Save parent identities and inversion signs.

Old explicit finite-path works can be re-scored using the uniform-sign auxiliary:
`W_plus = W_raw + (E_plus - E_raw)/kT`. This does not create weights for the
new implicit FM64 source. Three-atom geometry alone cannot diagnose parity
failure because triangles are planar.

## Completed evidence and its limits

| Evidence | Verified result | Interpretation |
| --- | --- | --- |
| `parity_refinement_rescore_v1.json` | N8, two trained streams, 2048 shared parents: convex minus affine -0.013366 +/- 0.002517 and -0.012610 +/- 0.002150 nat; convex minus typed -0.048981 +/- 0.017293 and -0.038923 +/- 0.016395 | Small one-condition relative-KL signal. Errors are row SEM conditional on trained models, not training-seed uncertainty. |
| Same N8 re-score | Path ESS remains approximately 1/2048 in every arm | Symmetry repair did not solve weight collapse or demonstrate distribution coverage. |
| `affine_context_xtb_v1.json` | 64/64 affine relaxations; medians 0.593530/0.591223 eV, close to convex | No meaningful measured geometry superiority. Old raw-source geometry panels remain separate. |
| `raw_oracle_symmetry_audit_v1.json` | Mirror energy defects 0.0533–0.2262 eV per condition | Motivated explicit source/target repair; retain raw results. |
| `even_oracle_contract_v1.json` | 8/8 conditions pass projected symmetry/force checks, 640 raw queries | Numerical interface qualification, not global potential accuracy. |
| `broad_source_complete_v2.json` | 8/8 source pools; 4096 train + 512 development parents each; 36864 raw queries | Exact source replay passes. Cached raw energy is not already E_plus. |
| `parity_entropy_smoke_v1.json` | 24/24 real engineering arms, 2304 raw queries | Training, inverse, intrinsic Jacobian and loader replay qualification. Only two optimizer steps. |
| `parity_smoke_replay_audit_v2.json` | New comparison script replays all 24 completed engineering arms | No production-performance result. v1 is retained as earlier audit. |
| `runs/verification/parity_training_full_tests.log` | 294 tests passed | Full suite at pre-handoff checkpoint. Assessment changes additionally passed 3 focused tests. |

Evidence filenames in this table are under `research/evidence/` unless a full
relative path is shown. The two N8 streams share evaluation rows and training
pool; never report them as 4096 independent evaluation samples.

## Live jobs: preserve and monitor

Verified at handoff: **45914819 and 45914826 are PENDING**. Scheduler forecasts
are mutable; no production arm has completed. See timestamped state JSON.

| Job | Scope | Limit / output |
| --- | --- | --- |
| 45914819 | Condition 0, six arms | 4 h; `runs/parity_entropy_condition_00_v1` |
| 45914826 | Array conditions 1–7, concurrency 2 | 4 h each except task 7 at 6 h; `runs/parity_entropy_conditions_1_7_v1/condition_XX` |
| 45916325 = 45914826_7 | Materialized task 7 | Same campaign, not an extra experiment |

Both jobs execute committed snapshot
`41516838cc13da4ffce9097404f95e3193422640`. Current working-tree edits do not alter
those jobs. `research/jobs.jsonl` records submissions and task-7 resource amendment.
Old raw-target jobs **45892106 / 45892107 were cancelled before execution**;
never resume or relabel them. Other projects have live jobs under this user;
do not cancel unrelated work to make room.

Production: 8 conditions x 3 methods x 2 training streams = 48 arms. Each uses
1000 updates, batch 16, 512 shared evaluation parents, 18048 raw queries
(16000 training + 1024 base evaluation + 1024 adapted evaluation).
Planned adapter total is 866304 queries, not completed compute. Source,
pretraining, failed branches and independent assessments cost additional work.

Run directories below each condition parent are `condition_XX_METHOD_sR`,
METHOD in convex/affine/typed, R in 0/1. A `panel.json` records terminal arm
outcomes including failures; `results.json` and loader qualification certify
individual completed artifacts. Seeds and protocol hashes are frozen.

## Immediate execution runbook

### 1. Refresh state and audit completed immutable arms

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR=/tmp/bgfm_mpl
export XDG_CACHE_HOME=/tmp/bgfm_cache
squeue -j 45914819,45914826 -o '%.18i %.9T %.12M %.30R'
sacct -j 45914819,45914826 --format=JobID,State,Elapsed,ExitCode -P
BGFM_AUDIT_DIR=$(mktemp -d "$PWD/runs/verification/claude_campaign_XXXXXX")
envs/flowmol/bin/python -u scripts/research/compare_parity_campaign.py \
  --runs-root runs --out "$BGFM_AUDIT_DIR/campaign.json"
```

The comparison script now exists and has passed the real 24-arm engineering
replay. It checks protocol/artifact hashes, frozen recipe, saved inversion
signs, checkpoint reconstruction, log volume and matched contrasts. It emits
complete/failed/**unresolved** counts; a missing terminal entry is not proof of
failure or success. Add `--require-complete` only after all prescribed arms
have terminal ledgers. Report both absolute nat and per-internal-DOF contrasts,
per condition and stream. Do not pool shared rows across streams as independent.

If a job times out, first inventory completed arms and partial files. The panel
runner refuses an existing `panel.json`; it is **not a resumable launcher**.
Implement a versioned missing-arm recovery with explicit parent links, retaining
the original failed allocation and query lower bounds. Do not delete its ledger
or rerun all six arms into the same output directory. Commit before using
`scripts/research/submit.py` to produce new immutable source snapshots.

### 2. Independent geometry and diversity assessment

`scripts/research/assess_work_panel.py` now supports strict
`--require-shared-parents`, explicit development-manifest identities and fast
pair-distance profiles. It accepts samples without work; do not fabricate ESS.
Broad assessment orchestration and cross-condition aggregation still need to
be implemented. Qualify on one completed condition before submitting all eight.

Inside a CPU allocation, with FlowMol's `bin` on PATH and `lib` on
LD_LIBRARY_PATH (see `scripts/research/affine_context_assessment.slurm`), an
example for a completed condition is:

```bash
BGFM_CONDITION_DIR="$PWD/runs/parity_entropy_condition_00_v1"
BGFM_GEOMETRY_DIR=$(mktemp -d "$PWD/runs/claude_geometry_00_XXXXXX")
envs/flowmol/bin/python -u scripts/research/assess_work_panel.py \
  --samples base "$BGFM_CONDITION_DIR/condition_00_convex_s0/base_samples.pt" \
  --samples convex "$BGFM_CONDITION_DIR/condition_00_convex_s0/adapted_samples.pt" \
  --samples affine "$BGFM_CONDITION_DIR/condition_00_affine_s0/adapted_samples.pt" \
  --samples typed "$BGFM_CONDITION_DIR/condition_00_typed_s0/adapted_samples.pt" \
  --out "$BGFM_GEOMETRY_DIR" --require-shared-parents \
  --xtb-count 32 --seed 9059 --workers 4 --max-cycles 200
```

Retain both training streams, the same seeded parent subset, failed arms and
failed relaxations. Report convergence denominators, contacts, diversity and
strain jointly. A contact graph or successful xTB relaxation is not complete
chemical validity or thermodynamic mode coverage. Use the corrected saved
inverted outputs; do not silently reuse old raw-source geometry results.

### 3. Qualify external EACF/FAB and matched HMC

Official EACF source is unmodified at
`/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_20260910`, commit
`beafab1b1ccd2b770572daeef1cf15f3fe199c21` (MIT).
Separate environment:
`/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_env_20260910`.
Dependencies installed; core imports and `pip check` pass. The installer is
finished. **No EACF model, energy bridge, GPU execution or baseline result is
qualified yet.** Read `research/evidence/eacf_runtime_install_v1.json` for the
actual versions and retained installation failures.

The resolved wheel is `jaxlib 0.4.13+cuda12.cudnn89`, despite the install log's
historical CPU name. Imports were forced to CPU; CUDA is untested. Chex 0.1.8
is yanked upstream for outdated requirements, recorded rather than concealed.
No FAB-JAX or Torch runtime has yet been installed in this separate environment.
Do not modify the existing FlowMol/fairchem environments to satisfy EACF.

Next: qualify a bounded upstream model test, e.g.
`eacf/flow/aug_flow_dist_test.py::test_distribution`, with explicit CPU resources
and caches outside home; inspect its dependencies and runtime before scaling.
Then qualify E_plus energy/force bridge and charged/open-shell conditions, pin
FAB dependencies, and freeze the matched-compute comparison. Default upstream
reflection behavior must be inspected for the projected target. Disable external
experiment logging. A joint EACF KL bound or joint ESS is not our exact marginal
KL change or marginal ESS; do not rank unlike quantities. HMC/MH must use E_plus
and retain all raw-query, equilibration, correlation and failure costs.

### 4. Resolve scientific gates before final claims

The next decision is whether the current candidate's gain survives the eight
prespecified conditions and same-context affine control with meaningful effect
size, acceptable geometry/diversity and fair compute. A tiny significant change
alone is insufficient. Do not tune on the 512-row reporting panels or choose a
best checkpoint after seeing results. If results are mixed or negative, report
all of them and formulate one bounded follow-up addressing the measured failure.
Do not reopen every failed branch or invent a new critic while evidence is pending.

Strong external comparisons and credible independent distribution checks remain
necessary. Energy distributions include density of states; lower energy alone
is not correct probability mass. Remaining N8 ESS collapse is negative evidence.
An odd-energy control variate is only a prospective variance-reduction idea;
it is not implemented or an established contribution.

**722 reserved conditions remain untouched.** Freeze a defensible final protocol
only after development decisions, then evaluate without tuning on reserved
outcomes. Audit metadata/source overlap limitations; record-level links do not
exhaust all trajectory relationships. Do not describe the inherited pretrained
checkpoint as comprehensively held out without evidence.

### 5. Finish the actual manuscript and reproducibility package

Edit `paper/main.tex` and the M1/M2 sections. Latest verified draft:
`runs/verification/paper_20260910_parity_final/main.pdf`, six main pages.
`paper/legacy_audit.tex` is historical, not the current contribution.
Use `bash paper/build.sh <new-output-directory>` and preserve old builds.

Update tables from audited artifacts, separate raw and projected protocols,
include external baselines, effect sizes, failures, compute, limitations and
implementation details needed for reproduction. Recheck prior art using primary
sources before claiming architecture novelty. Build an extracted reproducibility
package and verify it from that package, not only the working checkout.

Recorded planning dates in `research/PLAN.md` are abstract September 18 and
paper September 25, 2026 AoE (official guidelines last checked September 8).
Recheck the official ICLR page before submission planning. An internal evidence
decision is planned for September 15; a complete draft for September 20.
The old branch-specific actions in PLAN are historical; use this runbook/NEXT.

## Storage, environment and evidence discipline

- Never write under `/n/home04/yulili/`, including caches. Persist under this
  laboratory checkout or the external baseline directory; use /tmp for scratch.
- OMol25 primary; max_atoms 200; bond-free `total_loss_weights.e = 0`.
  Preserve inactive `projection.py`, non-finite loss guards and the legacy
  patch-based entrypoint's pop-bgfm-before-model_from_config order.
- Separate `envs/flowmol` (Torch 2.2/DGL) and `envs/omol25` (Torch 2.8/fairchem).
  Never install fairchem into FlowMol. Set PYTHONPATH to the active source or
  snapshot so editable installations do not silently select old code.
- Do not modify shared FlowMol under `/n/holylabs/woo_lab/Lab/yulili/bgfm/`.
  The eSEN checkpoint and original data really are available there; see
  `research/evidence/oracle_availability.json` and source protocol manifests.
- Raw energy RPCs use `evaluate_chunked(max_request=32)`. Failed bulk producer
  45885669 had an attempted-query bound 0–8192, not proven zero cost. Keep
  coordinates, scored chunks, acknowledgements and lower bounds on failure.
- Preserve failed legacy density/CNF, score/Stein, work, covariance,
  empirical-CFM and auxiliary branches. No old endpoint head is velocity by
  assumption; no test PASS is a Boltzmann or novelty certificate.
- Do not mark the full ICLR goal complete from this handoff. Leave a fresh
  NEXT/status note after meaningful work with exact job IDs, paths and failures.
