# Next research decision

Full ICLR goal remains active and unachieved. This goal turn is PROGRESS:
implemented generic broad-condition training, passed24 real-interface cases,
repaired a verified bulk-oracle timeout, released the complete matched campaign,
replayed both neural-context ablations and rewrote the primary manuscript around
our current method.294 tests pass. No broad calibrated sampler or ICLR-ready
contribution is established. Do not mark complete from code or format checks.

LATEST: projected-potential audit45906924 PASSES all8 conditions/640 queries.
New-source/target N8 re-score45906933 completes14336 queries: convex-minus-affine
-.013366+/-.002517 and-.012610+/-.002150 nat; all path ESS remainabout1/2048.
Corrected training interface45911477 passes24/24 cases,2304 raw queries.
All8 full raw source pools are complete (45889306,36864 queries).
The old held raw-target jobs45892106/45892107 were cancelled before execution.
NEW corrected full production names are parity_entropy_condition_00_v1 and
parity_entropy_conditions_1_7_v1; obtain fresh job IDs/state from jobs.jsonl.
Each full corrected arm uses18048 physical queries;48 planned arms866304.
Read the exact frozen parity_training_protocol_v1.json and its source-law scope.
Do not resume or relabel old raw-target jobs. No learned odd control variate
has been implemented. Full scientific readiness remains false.

CRITICAL NEW FINDING: raw eSEN is not inversion invariant on the32 checked
geometries. Per-condition mirror differences reach .0533--.2262 eV, while proper
rotation/permutation errors are ~1e-6 eV. The source FlowMol config ALSO uses
n_cp_feats=4 and cross products, so DO NOT assume q0 is O(3)-invariant merely
because our adapter is. Raw-target relative-KL results remain raw-target results.
Read notes/parity_refinement_candidate.md before continuing.

Immediate actions:
-45892106 and45892107 were HELD before execution for source/target review.
 Do not unhold their immutable raw-target scripts under a new interpretation.
-45889306 source production continues: coordinates and raw energies remain
 useful inputs/components for a new protocol. Re-query the SAME handle.
-Projected-potential audit even_oracle_contract_v1 and old-N8 re-score
 parity_refinement_rescore_v1 are submitted; obtain their job IDs from jobs.jsonl.
 The first uses a larger FD ladder; the second adds inverted energies to all
 seven stored2048-parent arms. Inspect terminal results and preserve all failures.
-Source/target repair implementation: parity_refinement.py. Set
 q0_plus=(q0+inversion#q0)/2 and E_plus=(E(x)+E(-x))/2; forces are
 (F(x)-F(-x))/2. Inference uses one independent recorded sign per parent.
 Exact transport-KL change compares T#q0_plus to q0_plus; its even bracket can
 be evaluated on original parents. The separate source augmentation gain is
 JS(q0,inversion#q0) in[0,log2], not zero entropy change.
-Old explicit works can be re-scored with a specified uniform-sign auxiliary:
 W_plus=W_raw+(E_plus-E_raw)/kT. Uniform proposal/target sign factors cancel;
 this does not require the original source or reverse path to be parity invariant.
-294 tests pass, including4 new symmetry-mixture/force/work tests and explicit
 inversion tests for both neural families. Real molecular projection now passes its prescribed checks. Do not claim the parity issue explains the entire ESS failure.
-Freeze a NEW broad training source/target and matched physical-query recipe
 after these checks. Raw one-query gradients are unbiased for E_plus only with
 properly randomized source inversion. Paired gradients need both orientations
 and half as many parents at the same oracle budget. Independent evaluation
 uses exact E_plus. No learned odd-energy control variate exists yet; it is a
 prospective idea only if measured gradient variance supports it.

Earlier campaign map (partially superseded by the parity review):

1. Re-query the SAME active campaign handles before acting:
   -45889306 species_breadth_source_full_v2: regular GPU,4-hour cap, all eight
    conditions sequentially. Source0 is complete; later conditions are ongoing.
   -45892106 broad_entropy_condition_00_v1: six trained arms for condition0;
    was PENDING for priority at the last check. Do not duplicate it.
   -45892107 broad_entropy_conditions_1_7_v1: array1--7, concurrency2, afterok
    dependency on45889306,4 hours per task. Validate source records inside each
    task; scheduler completion alone does not qualify a condition.
   Each full condition has4096 source-training and512 development rows. Keep
   every source failure and arm failure. The source has no evaluated path weights.
2. The generic trainer IS implemented: train_broad_entropy_adapter.py reads
   entropy_source.py's strict finite_fm_gaussian contract. It compares convex,
   same-neural-context affine and typed-linear at1000 steps/B16, with init9161/
   9162 and selection9141/9142. Each arm17024 physical queries;48 arms planned,
   total817152, plus36864 source queries and other costs. Report actual finished
   costs separately from budgets. Tests include corrupt electronic state, stream
   seeds, COM, energy values, invented weights, target and runtime metadata.
3. Aggregate completed production arms with source/target hashes, paired512-row
   KL changes, per-internal-DOF values and all failures. Parent directories:
   runs/broad_entropy_condition_00_v1/condition_00_METHOD_sREPLICA;
   runs/broad_entropy_conditions_1_7_v1/condition_XX/condition_XX_METHOD_sREPLICA.
   Methods convex/affine/typed, replicas0/1. Do not pool seeds as independent
   sample rows. Add independent geometry/diversity/xTB assessment, retaining all
   denominators. assess_work_panel.py already supports absent work, but its
   validation_index fallback must explicitly use the new manifest identity
   before multi-condition aggregation. Broad performance is not available yet.
4. Both old N8 affine-context comparisons are now complete and replay-verified.
   Convex-minus-affine on the shared2048-row panel is-.01332+/-.00252 and
   -.01259+/-.00215 nat. This is a small one-condition, two-stream component
   advantage. Path ESS remains about1/2048. Evidence:
   species_affine_replication_audit_v1.json, with all coords/volumes/work replayed.
   Its affine-only xTB launcher is affine_context_assessment.slurm; inspect
   jobs.jsonl to see whether submitted, and reuse existing convex xTB rows
   when comparing so previous energy/relaxation calls are not silently repeated.
5. Source45885669(v1) was CANCELLED at14m12s after a4096-row CPU oracle RPC
   exceeded its60-second reply limit; neither started condition qualified.
   Preserve source_timeout_v1 evidence: actual attempted count0--8192, not zero
   physical cost. Replacement uses evaluate_chunked(max_request=32), persistent
   coordinate/scored chunks, and acknowledged/requested counters. Real check
   45888793 reproduced64 cached energies each on conditions0/7 exactly, with
   32-row calls taking1.14--4.15s. Full24-arm smoke45888794 passes1536 queries.
6. External EACF baseline source is cloned, unmodified, at
   /n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_20260910,
   commit beafab1b1ccd2b770572daeef1cf15f3fe199c21 (MIT). No runtime/environment
   has been installed or model qualified. Read notes/eacf_comparison_contract.md
   and eacf_upstream_inventory_v1.json. Upstream used Python3.10/JAX0.4.13 and
   CPU Torch; keep any new environment separate from FlowMol and fairchem.
   EACF energy experiments use FAB; simply comparing its joint-KL bound with our
   marginal-KL change is invalid. Strong external flow and matched HMC controls,
   reliable independent distribution references and full compute remain required.
7. The PRIMARY manuscript has been rebuilt around exact-entropy refinement:
   paper/main.tex, sections/M1_refinement.tex and M2_refinement_proofs.tex.
   Latest build: runs/verification/paper_20260910_refinement_v4/main.pdf.
   Main text5/9 pages; broader/external evidence remains explicitly incomplete.
   Original audit entrypoint is paper/legacy_audit.tex; its sections and old
   PDFs are preserved. build.sh accepts an optional second entrypoint argument.
   Do not return to appending all new work only to the old audit appendix.
   Update the new main text once experiments and scientific gates justify it.

Reserved722 outcomes remain untouched. Keep all score/CNF/work/covariance/CFM/
auxiliary failures. Preserve original charge/spin, bond-free OMol25, max_atoms200,
immutable source snapshots and the two existing environments. Never write home
or alter shared FlowMol. Final independent evaluation and an author-ready
reproducible package remain required for the full objective.
