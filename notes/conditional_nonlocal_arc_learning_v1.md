# Conditional nonlocal energy learning

The audited physical site-arc chain is the current strong proposal baseline.
A local vector score fitted near one direction is not sufficient: its global
extension can leave support, and higher one-step acceptance did not reliably
predict complete-chain progress. This round predicts conditional energy changes
across the actual feasible angular domain, with one frozen parent split.

`ConditionalArcEnergy` encodes all passive geometry, atom types, the desired graph,
root radius and original charge/spin/temperature. The root direction is removed
before message passing. A small context network predicts bounded coefficients
of radial basis functions of each candidate root-to-passive distance, normalized
by covalent-radius sums. Query energies are sums of these context-conditioned
interaction curves plus the physical site score converted to energy and the
analytic COM confinement contribution. No absolute energy offset is learned.
The scalar is invariant under O(3), translation and joint atom reordering, and
its tangent derivative provides a conditional force prediction. The representation
is a standard conditional neural energy construction, not a new physical law.

At inference the actual log score is minus surrogate energy divided by kT.
The circle code evaluates this score at every interpolation endpoint, retaining
exact segment masses, both oriented-circle preimages and their source-dependent
normalizers. Generic scalar-score support is implemented and tested. The new
model is now integrated through the separate `arc_energy` complete joint decoder.
Every root/order uses its actual context, with independent NumPy energy readout
and density quadrature checks. The known joint-target check exercises accepted
movement and preserves its tested observables. Do not pass the scalar model
through the old one-vector `arc_model` interface.

The dataset contains438 audited contexts:206 FIT,76 withheld-parent and156
withheld-composition contexts. Every context contains its center, all supported
local probes and four frozen valid arc endpoints. For FIT parents, the two local
check directions stay excluded from fitting. All withheld parent/composition
labels are excluded. The added928 withheld endpoints cost1,856 raw calls and
have their own fixed protocol and full oracle replay. These are internal training
validation data, not the722 reserved final outcomes.

The first training comparison uses identical architecture, initialization and
parent-balanced batches for work-only and work-plus-force losses. Work is a
potential difference within one fixed context; force is the tangential derivative
of that same conditional energy, in eV. Both use a0.1 eV-scaled Huber error; the
force term has weight0.1. Two seeds each run800 fixed optimizer steps. Checkpoint
curves are diagnostics; the final checkpoint is evaluated without selecting a
favorable intermediate step. No fresh oracle queries are required for training.

Evaluate local-check and nonlocal-arc work error and tangential-gradient error,
with parent then composition averaging, against both the untrained site-plus-
confinement initialization and plain site64. A fit improvement alone is not a
successful learner. Internal work prediction cannot establish sampling efficiency:
a promising frozen model must still beat the stronger physical arc baseline in
complete-chain experiments, with all preparation, probe, fitting and inference
costs. Any failed objective/seed and adverse held-out result remains recorded.

The first four trainings and their independent audit are complete. All438 data
contexts and split masks are rebuilt; every final metric and both physical-score
baselines are replayed. Actual trained-loss finite differences agree to1.19e-8.
Both objectives and seeds improve nonlocal work prediction on withheld parents
and compositions (about19--24%). The added force loss has no established decisive
advantage over work alone. No learned complete-chain result is available yet.
See `research/CONDITIONAL_NONLOCAL_STATE_20260912.json` for exact values and costs.
