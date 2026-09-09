# Next research decision

The user's explicit ICLR objective is active. It is not achieved. See STATUS
and the evidence files before running more experiments; do not restart the audit.

1. Audit the newly recovered official validation archive (2,762,021 records),
   preserving source IDs, charge/spin, and precision. Check training overlap and
   establish an evaluation manifest before querying final method outcomes.
   Path: woo_lab bgfm/raw_data/omol25/v250514/official_validation/val_extracted/val.
   SHA256 of the compressed archive:
   bc2f2aa70459ed029217f188d1e8ab0d90abbb90065235f6414d79246bad1ab8.
2. Summarize the completed center-refinement and matched-budget IS experiments.
   The eight-atom condition remains low-ESS; refining centers alone is not a
   demonstrated solution. Count the 1,632 pilot oracle queries when comparing
   three 1,024-particle repetitions with three 1,568-particle controls.
3. The independent AgBr2 reference is complete: log(mean Z-hat)=144093.1275689,
   relative SE about 1.67%. The simple confinement Gaussian beats both learned
   template proposals at 1,024 direct-IS queries per seed. Preserve this baseline.
   Do not promote symmetry averaging or defensive mixtures as new mathematics.
4. Electronic-state conditioning is now implemented and trained. Both 10k-update
   continuations pass 32/32 xTB, but global-state median strain 3.94384 eV does
   not improve on legacy continuation 3.89309 eV. This remains a necessary
   semantic correction, not a positive generation result. Sampling-panel loading
   and xTB use the recorded original multiplicity and prior width.
5. Before physics-student training, choose a teacher whose normalizer/moments
   and sampling coverage survive independent checks. A possible next controlled
   comparison is mixing exact independence-MH refreshes with local MALA inside
   SMC, since all current SMC mutation is local. This is established MCMC. The control is now implemented and running in
   job 45706103 with matched potential budgets; inspect completed results before
   launching further variants.
6. Temperature/constraint choices are explicit computational targets. Current
   kT=1 eV is not ambient-temperature molecular equilibrium. A prospective
   physically motivated temperature study must retain existing negative results
   and freeze its protocol before use of the new evaluation data.

Raw training replay is complete and exact: 3,941,522 records. Use the checked
`source_index_readonly.sqlite` with immutable read mode. The original producer
used WAL and caused a cross-host live-reader locking error; it completed normally.
Future producers use DELETE journaling. Read the NumPy sidecars through
ElectronicMetadata, which verifies completeness and input hashes.

Current tests: 177 pass. Main-text build: 9/9 pages. Both are engineering gates,
not evidence of ICLR readiness. The original main paper remains an audit /
development draft; rewrite around a genuine contribution only once supported.

The old perturbation-shard draft is archived at
notes/archive/build_perturbation_shard_unvalidated.py and was never used in the
reported experiments. No subagents, external publication or messages to
other people have been authorized. Author and submission management belongs to
the user. Never write home or edit the shared FlowMol installation.


Official validation audit job 45706102 and global-refresh job 45706103 were
confirmed RUNNING during this continuation. Inspect their live scheduler state
and immutable run directories before relying on old status. Reserved evaluation
conditions have not been subjected to method outcome queries. RegFlow must be
considered alongside FALCON if an exact-likelihood student is investigated.


Latest continuation: official-data audit and global-refresh SMC are complete.
Use committed `official_development_candidates.json` for method development;
`official_reserved_candidates.json` must remain untouched by method outcomes
until a protocol is frozen. The new-data audit has 2,564,135 eligible records and
zero checked old-composition/source-link overlaps; candidate sets are 664/722.

Global refresh did not solve the eight-atom problem. A mean-work-trained
stochastic teacher is now implemented with an exact Gaussian reference bridge,
full checkpointed gradients and a first-order-only external-force handoff.
Do not confuse this with a final FM student. Initial geometries are still poor;
the joint/backward-only GPU calibration must be inspected before larger runs.
The old Euler preflight's parameter-gradient check passed, but its geometry
was bad; its source commit and negative results are retained. For any new
architecture, account for RegFlow, FALCON, SNF and the other prior work.


Current GPU calibration jobs: 45716497 (`mean_work_joint_5846_v1`) and
45716504 (`mean_work_backward_5846_v1`), source 87f4df7. Each is bounded to
two GPU-hours and 100 updates. Inspect exact scheduler/run state before
reporting them active or complete. Training uses native prior width one and
exact Gaussian-reference transitions to the confinement width; old Euler
preflights are retained separately and were geometrically poor.
