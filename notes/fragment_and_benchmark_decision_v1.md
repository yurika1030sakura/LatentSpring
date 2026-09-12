# Fragment and generator-baseline checkpoint — September12

Read `research/FRAGMENT_AND_BENCHMARK_STATE_20260912.json` for hashes and the
current state. This turn is progress; the scientific ICLR goal remains active.
Apply the user's clarified standard in `research/CLAIM_AND_BENCHMARK_SCOPE.md`:
a scoped, useful contribution with sound evidence, not a nearly perfect generator
or universal physical dominance. A new physics law is not required.

## Completed physical construction

`cfm_mol/fragment_exchange.py` and `fragment_sampler.py` implement pendant-fragment
attachment exchanges, normalized root-vector proposals, uniform torsions, a full
augmented inverse and the appropriate MH correction. Internal geometry is retained
while the root vectors exchange with auxiliaries. The augmented Jacobian is one;
root densities retain r^-3. The code rejects the near-antipodal alignment case
symmetrically and keeps all actual inverse action counts.

Six targeted tests pass, including the full COM-plus-eight-auxiliary Jacobian,
unequal fragment sizes/singleton limit, rotations/reflections/permutations,
known Gaussian target with paired action labels, missing-density-ratio negative
control, collective force-response differentiation and physical-query accounting.

The frozen eight-composition geometry screen identifies supported multiatom
backbone changes in conditions1/2/4/6. This is structural eligibility, not energy
acceptance. Conditions3/5/7 retain zero support under the current validator.

## Physical pilot and preserved failure

The original job46122960 completes condition1 and aborts during the existing
force-rotation step of the Al-containing condition2 local control. RDKit raises
`IndexError: unordered_map::at`; the failed arm uses650 acknowledged/requested
raw calls. The dependent audit46123304 is cancelled. Do not convert this error
into an ordinary chemical rejection or silently redefine the target.

Job46124570 continues the previously unstarted conditions4/6 under the original
protocol and source. Audit46125685 replays all completed arms, all physical pairs
and RNGs, and reproduces the exact failed Al prefix/exception with the failing
geometry saved. Results:18 completed evaluation arms,1 failed,5 Al arms unstarted,
and4 completed warm arms. Total new raw physical calls18732, including the failure.
All job handles are terminal. The planned four-condition panel is not complete.

The transition census inspects ALL kernels, including local moves, rather than
assuming local moves cannot change the perceived graph. Counts of accepted
non-H/non-halogen backbone-adjacency changes accompanied by canonical-SMILES
changes are:

| Condition | Local control | All singletons | Fragments, cap4 |
|---|---|---|---|
|1:12 atoms, neutral triplet |0/0 |0/0 |2/2 |
|4:17 atoms, neutral singlet |0/0 |0/0 |0/1 |
|6:14 atoms, charge+1 singlet |0/0 |1/1 |3/2 |

Each cell is two seeds, four generated starts and128 microsteps. Actual query
counts differ because invalid proposals skip the oracle. These are perceived
graph-transition diagnostics, not reaction rates, independent molecular discoveries,
equilibrium qualification or AI superiority. No fragment-specific learner exists yet.

## Generator comparisons now included

Job46126332 applies a common structural readout to all512 matched development
parents of the available frozen generator baselines, with no new physical calls.
Graph-supported output counts for two replicas are FM64(4,4), previous convex
adapter(4,5), published EACF(15,14), and compact EACF(4,6). EACF source coordinates,
parent IDs, inversion signs, checkpoint/audit hashes and original targets are
checked. These are existing generator outputs; no old KL or target is relabelled.
The absolute support rates are poor on this composition and must remain visible.

This pilot is not the final end-to-end comparison with the current MCMC method.
It covers one composition and does not make correlated chain snapshots equivalent
to independent generator parents. Keep training, inference, realistic amortization,
source attempts and failure/correlation conventions explicit.

## Next concrete action

Implement the direction-augmented EACF MH baseline in
`research/CLAIM_AND_BENCHMARK_SCOPE.md`, checking the known-target kernel and real
upstream interface before molecular use. Source inspection verified:

- Isolated environment `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_env_20260910`;
  upstream `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/external/eacf_20260910`,
  pinned commit `beafab1b1ccd2b770572daeef1cf15f3fe199c21`.
- `adapter.pkl` stores `theta`, `recipe` and condition; `build_reference_flow`
  recreates the architecture. Forward and inverse interfaces are
  `bijector_forward_and_log_det_with_extra_apply` and its inverse counterpart.
- `FullGraphSample.positions` stacks physical x and auxiliary a at axis-2.
  Features broadcast atomic numbers to [batch,N,1,1]. Physical x lies in COM
  dimension3(N-1); the one auxiliary point cloud has3N unconstrained coordinates.
- The audited auxiliary target is Gaussian with mean x and `recipe['aux_scale']`.
  `audit_eacf_joint_refinement.py` already checks its analytic density and replays
  all512 old forward/inverse samples. Use the known auxiliary density, not the
  unknown FM source likelihood.

This comparable learned-sampler baseline is not implemented yet. After it is
validated, freeze representative generation/corrector and sampling comparisons
with measured reuse costs. Do not add more singleton-mixture complexity simply
because it is easy. Keep the collective fragment-force identities available for
later learning, and qualify cheaper geometry-only preparation under a NEW protocol
without erasing historical costs.

The working paper is eight main-text pages and contains the new fragment proof,
physical results, preserved Al failure and generator-baseline diagnostic. It is
still a development manuscript; scientific submission readiness remains false.
