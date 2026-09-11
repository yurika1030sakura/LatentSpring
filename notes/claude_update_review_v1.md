# Review of the Claude update after the first handoff

The preserved starting point is commit e88c1d8 plus the then-uncommitted files
in `runs/claude_update_audit_v1/source/`; `manifest.json` records their hashes.
This is an observed source snapshot, not proof of code loaded by earlier jobs.
The executable evidence is `research/evidence/claude_update_review_v1.json`.

## What helps, and what does not yet establish a stronger method

The homogeneous-context diagnosis is useful: replacing every same-element atom
by the common centroid leaves no directional context on a homogeneous system.
The node-conditioned part is inactive, although the radial map still depends on
the active coordinates. Describing the entire transform as having no
configuration dependence was too strong. The original architecture's known
homogeneous expressivity limitation is real.

Fixed index halves supply nonzero conditioning vectors but lose same-element
permutation equivariance. With fixed seed771 and nonzero trained-style heads,
swapping an atom across halves changes the expected equivariance relation by
0.3693, 0.2962 and 0.2159 coordinate units on N4, N8 and N13 respectively.
The corresponding log volumes also change. Both half-centroids remain exactly
fixed, leaving three collective coordinates unadapted on a homogeneous cluster.
Stacking the same split for16 sweeps does not remove that invariant.

COM preservation and an exact conditional determinant still hold; full intrinsic
Jacobian and inverse tests pass. Thus this is a legitimate *labelled transport
diagnostic*, not a qualified permutation-equivariant repair. `split_groups=False`
remains the default and retains archived whole-element checkpoint behavior.
Configuration now states its permutation contract, and loader decoding validates
derived layout metadata. No random-permutation mixture has been introduced:
such a mixture would need its own density/entropy accounting.

The reference EACF construction uses auxiliary coordinate splits to retain
permutation equivariance. Its joint and marginal likelihoods must still be
distinguished in comparison. See the
[primary paper](https://arxiv.org/html/2308.10364v6) and our comparison contract.
Do not claim that fixed index splitting solves the same architectural problem.

## Reproducibility failures repaired

Three1000-update molecular jobs45931183/45931185/45931191 finished their oracle
work, then failed because `minimum_active` and `internal_blocks` were passed as
constructor arguments. A strict decoder now separates constructor options from
validated derived metadata. The first historical split format can be identified
from its stored blocks; unknown fields or inconsistent layouts fail closed.

All512 saved parents, positions, log volumes and endpoint-KL changes in each
of those three jobs replay in `runs/anneal_loader_recovery_v1`. Position errors
are at most3.11e-15 A. The original FAILED allocations and incomplete reports
remain untouched. Recovery records link their original artifacts/logs and report
zero new oracle queries. Their original54144 acknowledged queries still count.
Earlier fast failures45928976/45928981/45928982 are separate, retained failures.

The two LJ13 arrays45934998 and45935775 were live concurrently and wrote the
same12 filenames in `runs/particle_bench_v1`. Their archived batch scripts and
task0 logs prove shared output names and different first objectives. Both arrays
were cancelled after source/log/result snapshots were saved. They are not two
independent replicas; their combined result files cannot certify a target or
training trajectory. The earlier wrong-oscillator branch and DW4 outcomes remain
preserved. Historical particle runs did not save model checkpoints or sampled
positions, so their scalar ESS cannot be independently replayed.

Future launchers require a committed source snapshot and a unique campaign root.
Particle output creation is exclusive and fails before a duplicate run can write.
New diagnostic runs save model/optimizer checkpoints, evaluation parents and
samples, source hashes, query counts and replay verification. Evaluation is
chunked. Two bounded two-update DW4/LJ13 engineering checks pass; they are not
performance experiments. The full repository suite passes310 tests.

## Particle target and estimator scope

The checked reference is the pinned EACF source at
beafab1b1ccd2b770572daeef1cf15f3fe199c21, particularly
`target_energy/double_well.py` and `target_energy/leonard_jones.py`.
The unordered pair convention removes the reference's factor1/2. LJ's harmonic
coefficient0.5 lies outside the pair epsilon/(2*tau) prefactor. DW4 here uses
reference tau1; the upstream example also supports another temperature, so
benchmark names alone do not establish identical targets.

The undocumented distance smoothing and1e-4 LJ floor were removed from future
reference evaluation. Exact coincident particles have infinite LJ energy and
zero target weight. Undefined/all-zero weights and nonfinite gradients must
fail explicitly. Annealing divides the full training reduced energy; it is a
training path, not a claim that all history values use the physical temperature.
The last update now reaches the final temperature even at anneal_fraction=1.

Exact sample density ratios do not make Monte Carlo ESS or energy means exact
population quantities. Near a pair collision in three dimensions, a smooth
positive proposal density contributes an energy integral proportional to
integral r^(-12) r^2 dr, which diverges. Consequently, an unsoftened LJ reverse-KL
population objective requires an integrability argument; finite minibatch losses
do not supply it. This does not invalidate finite-sample importance weights,
but their ESS is not a missing-mode or finite-KL certificate. No new regularized
reference target or energy-clipping training objective has been silently chosen.

The approximately114-nat log mean path weight from the old N8 experiment is not
an absolute source-target KL estimate, especially with path ESS about1/2048.
It cannot be compared to a reachable log-volume range as a measured entropy
capacity requirement. Removed that premise from annealing documentation.

## Actual molecular evidence now available

The original corrected48-arm campaign remains on its immutable41516838 snapshot.
Condition0 completes all six arms and108288 acknowledged queries. A current
13-arm immutable-prefix replay passes. On condition0, convex minus same-context
affine is -1.6714 +/-0.5352 and -1.5623 +/-0.5334 nat, and convex minus typed is
-5.9168 +/-0.7239 and -5.8568 +/-0.7006 nat. These are shared512-parent row SEMs,
not training-seed uncertainty or absolute KL estimates.

At that prefix, condition1 also favors convex over affine, but its first stream
does not show improvement over typed (+0.1646 +/-0.4234 nat). Condition7 has only
one completed convex/affine pair. Keep these mixed outcomes and all unfinished
arms in the denominator. Do not promote the first completed conditions into an
all-eight claim. New source weights and calibrated coverage remain unavailable.

Next actions: independently assess condition0 geometry/diversity/xTB on the same
32 seeded parents across base and six trained arms; complete and audit the full
molecular campaign; qualify EACF/FAB and matched HMC. Do not immediately relaunch
the large particle grid. A permutation-equivariant homogeneous extension, useful
external-baseline gains and credible coverage remain research requirements.
EACF dependencies have passed imports, but the full model/energy bridge has not
been qualified yet; this turn prioritized the observed correctness failures.

## Independent condition0 geometry follow-up

CPU job45938396 completes in55 seconds:224 attempted GFN2 assessments,192
converged and32 failed across base and six arms, with every failure retained.
Base converges29/32 with median strain5.1851 eV; convex streams converge28/32
and29/32 with medians4.6595 and4.6550 eV. Affine controls converge27/32 each,
with medians4.7487 and4.8204 eV. These are success-only medians; the paired
convex-versus-affine ranking favors convex20:10 and21:9, with2 pairs failing in
both arms each. This is a modest one-condition geometry signal, not full
chemical validity or distribution coverage.

Evidence: `research/evidence/parity_geometry_condition_00_v1.json`. All source
sample hashes and the shared-parent contract were checked. Current work is on
`iclr2027-arch-fix`; earlier handoff state remains a historical snapshot.
