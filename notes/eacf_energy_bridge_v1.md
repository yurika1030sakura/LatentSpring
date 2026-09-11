# External EACF energy bridge qualification

The pinned upstream CPU tests passed in job45939182 (85 s allocation): its
one-layer augmented distribution test and a19-node spherical coupling component.
This is runtime qualification, not a full trained baseline. Evidence:
`research/evidence/eacf_upstream_smoke_v1.json`.

`NumpyEnergyOracle` uses the existing omol25 worker without importing Torch or
fairchem in JAX. Bounded requests and acknowledged/requested counters remain
separate, including failed responses. An explicit byte buffer drains diagnostic
lines before waiting on the descriptor, avoiding a hidden TextIO-buffer response.
A lock serializes each RPC. Tests verify chunk counts and nonzero failed cost.
The existing Torch client and immutable production snapshots are unchanged.

`make_even_log_target` explicitly queries both orientations and returns
log pi_plus = -(E_plus + restraint/2 ||Px||^2 - energy_zero)/kT, evaluated at Px.
Its derivative is P(F_plus-restraint*Px)/kT. A batch cotangent must be expanded
on both atom and Cartesian axes. The independently calculated polynomial oracle
has an odd raw-energy component, so tests catch an incorrect mirror-force sign,
missing COM projection, and batch broadcasting errors. Single, batched, vmapped
and jitted reverse differentiation pass in the pinned JAX0.4.13 environment.
Only first-order reverse derivatives are implemented; not force Hessians.

The external callback follows the primary
[JAX callback documentation](https://docs.jax.dev/en/latest/external-callbacks.html)
and the pinned EACF external-energy pattern. Its numerical return is deterministic.
JAX can elide or repeat pure callbacks, so instrumentation reports actual worker
executions after blocking, rather than assuming a static query count. No
callback execution guarantee or higher-order derivative claim is made.

The frozen real probe uses the first two saved original parents from conditions0
and7, including neutral singlet and charge+2 triplet. Source results/sample and
oracle/protocol hashes are checked before execution. It checks saved E_plus,
weighted batch gradients, inversion and a .003/.0015-A finite-difference ladder.
The declared force-slope tolerance is .005 eV/A +2% of the directional slope.
One minimal upstream EACF optimizer step on condition0 then checks a real energy
parameter gradient, inverse, inversion and joint atom/feature permutation.
Its expected total is64 raw calls; actual acknowledgements must match. A runtime
or callback PASS remains engineering evidence, not molecular performance.

For the eventual matched-source EACF adapter, retain source q0_plus(x) r(a),
with independent unit Gaussian r on3N auxiliary coordinates. The target is
pi_plus(x) r(a). For its bijection S, relative JOINT KL has integrand
Delta U/kT + [||a_final||^2-||a_initial||^2]/2 - logdet DS.
Unknown q0 entropy cancels in this joint change. The final auxiliary conditional
KL is nonnegative, so joint and physical marginal changes cannot be ranked as
identical quantities. Primary reference:
[EACF](https://arxiv.org/html/2308.10364v6), sections3.3 and B.7–B.10.

This joint adapter does not supply a tractable absolute density for the implicit
FM source. It therefore does not automatically enable FAB/AIS weights from that
source. A standalone explicit-density EACF/FAB comparison needs its own source/
pretraining protocol and full compute accounting. All methods must use actual
E_plus in exponentiated weights and MH acceptance. Physics-based geometry and
qualified independent population diagnostics remain necessary comparisons.

## Completed real qualification

Job45945638 passes64 raw calls, including a full real-energy gradient/update
on a9160-parameter one-layer EACF component and recipe-based checkpoint
reconstruction. Both physical states pass cached energy and finite-difference
checks. Original failures45942379/45944337 retain8/36 calls; total108 calls.
The initial reduced-unit comparison was overstrict: exact repeated oracle inputs
show force differences up to7.69e-6 eV/A. The revised weighted-VJP gate is
explicitly1e-5 eV/A and the independent finite-difference ladder is unchanged.
No full EACF performance experiment or qualified population comparison follows.
