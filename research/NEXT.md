# Next research decision

The user's ICLR objective is active and not achieved. Do not restart completed
data audits. STATUS and evidence retain failures as well as improvements.

1. Inspect the larger-batch mean-work experiment after its terminal state.
   Compare joint gradients against forward-energy-only gradients plus backward
   conditional likelihood, using identical initialization, noise, steps, oracle
   budget and evaluation panels. First batch-two joint calibration reduced mean
   energy by 13.6534 eV, but ESS stayed 3.485/64 and unweighted overlap worsened.
   Backward-only samples stayed bitwise unchanged. Neither is sampling success.
   Independent xTB on the initial/100-update panels now gives 0/32 and 6/32
   converged, with extensive SCC failures. All 64 geometries per arm have multiple
   contact components. Compare the pending same-condition pure FM control before
   blaming this on the pretrained generator or declaring work training useful.
   Apply the frozen 32-index xTB protocol to the completed 500-update arms too.
2. Repeat promising training across seeds and new development conditions before
   choosing a final recipe. The 664 audited development candidates have no checked
   old-composition/source-link overlap. Use condition-only graphs. Reserved 722
   conditions must remain free of method outcomes until the protocol is frozen.
   The manifest-based work trainer now passes a real two-update PbCl2 CPU smoke;
   it opens no reference-coordinate dataset. Condition graph edge ordering has
   been corrected and tested before any broader experiment used it.
3. Require normalizer/moment/coverage checks and independent potential evaluation.
   The AgBr2 reference gives log(mean Z-hat)=144093.1275689 with relative SE 1.67%;
   simple confinement-Gaussian IS beats learned templates. On the eight-atom
   condition, defensive mixtures, center optimization and global-refresh SMC all
   remain poor. Keep these controls and their construction budgets.
4. Distill a physics teacher to FM only after its benefit is established. Account
   for SNF/FEAT, FKC, EWFM, FALCON and RegFlow. Work identities, symmetry averaging,
   defensive IS and exact-likelihood regression alone are existing methods.
   No defensible new molecular contribution is established yet.

Raw training replay is complete: all 3,941,522 accepted tensors matched bitwise;
original charges, spins, source IDs and float64 energies are restored. Use the
integrity-checked source_index_readonly.sqlite in immutable read mode. The old
cross-host WAL error did not invalidate the producer. Official validation audit
completed 2,762,021 records, with 2,564,135 eligible. Explicit source links do not
exhaust every possible parent-trajectory relationship.

Current kT=1 eV with harmonic confinement is a declared computational target,
not ambient-temperature OMol equilibrium. Do not change it after seeing results
just to improve metrics. Electronic-state FM conditioning is a semantic repair,
not a demonstrated quality gain: both 10k continuations pass 32/32 xTB, with
median strain 3.94384 eV (global) versus 3.89309 eV (legacy).

190 tests pass. Batched oracle inference passed the serial energy/force check;
serial remains the default and potential counts still include every structure.
The main-text build remains 9/9 pages and is an audit/development draft. These
engineering checks are not ICLR readiness. Authors and submission belong to the
user. No subagents or external messages are authorized. Never write home or edit
the shared FlowMol installation.

See research/jobs.jsonl for source snapshots and job IDs; re-query Slurm before
reporting any job as running or complete.

Submitted from a76313a: 45726541 (mean_work_joint_b16_5846_v1) and 45726651
(mean_work_energy_b16_5846_v1). Both were RUNNING at the September 9 20:13 UTC
check, with two GPU-hours maximum each. Do not infer completion from this note.

Same-condition pure FM control: 45729734, work_fm_control_5846_v2, source 9f99c7c,
regular gpu partition, half-hour maximum; PENDING for priority at 20:32 UTC.
Version v1 was rejected by gpu_test's submit limit and has no job ID. The source
reference itself passes xTB with strain .89014 eV and no overlap, so poor work
samples cannot be excused solely by unsupported condition/electronic state.

At 20:34 UTC, 45729734 had started RUNNING. The 500-update independent assessment
is queued as 45729911 (mean_work_xtb_500_v1), source bf7669a, afterok dependencies
on 45726541 and 45726651. It compares initial/joint/energy-only samples on the
same 32 fixed indices, with all 256 geometries per arm entering diagnostics.
Do not resubmit accepted jobs. If a dependency fails, retain that result and
inspect the dependency before changing the assessment job.

FM control 45729734 is now COMPLETED: 29/32 and 30/32 xTB convergence for
midpoint-16/64, versus 0/32 for bridge initialization and 6/32 after 100 work
updates. One overlap per 64 FM samples; solver-coordinate drift can still reach
.66522 A. Initialization damage is now a concrete concern. A native-mean option
compensates reference expansion before the same residual bound; inspect its
two-update preflight and independent structure metrics before larger training.
The 500-update reference-mean comparison remains a required retained control.

Native-mean preflight: 45730876, native_mean_preflight_5846_v1, source a2f4a7b,
two updates / batch two / 16 transitions / 64 evaluation samples, serial oracle,
RUNNING at 20:43 UTC. Its independent assessment 45730934 is queued afterok,
native_mean_xtb_5846_v1. Inspect both exact results before claiming initialization
repair or starting longer native-mean training. The suite currently passes 190
tests. No final reserved condition is used by any of these jobs.
