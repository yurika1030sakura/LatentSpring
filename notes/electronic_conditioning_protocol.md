# Explicit electronic states for the geometry model

Exact raw-source replay matched all 3,941,522 accepted records: 3,902,107 train
and 39,415 validation. Original energies are restored as float64 sidecars,
alongside unclipped total charge, multiplicity and calculation source IDs.
The historical tensors are unchanged. There are 72,423 charges outside the old
[-2,3] feature range, 678,514 non-singlets and 403,895 states above the minimum
electron-parity multiplicity. A full nuclear-charge/electron-parity audit finds
zero inconsistent entries. Source IDs do not by themselves establish a new
chemical-identity or trajectory-independent split.

The new geometry-only runtime patch broadcasts total charge, spin multiplicity
and a requested energy scale into the scalar node embedding. The legacy
atom-zero total-charge marker is neutralized inside each field evaluation;
this prevents an arbitrary atom-numbering convention from carrying the total
electronic charge. The state is constant across every node of each molecule.
No state is guessed when metadata is missing. The frozen composition prior
is a separate old checkpoint and must not be reconstructed from this changed
position model.

The embedding uses charge/4, log multiplicity, log requested kT, log atom count
and (multiplicity-1)/atom count. Its final layer is zero-initialized. Charge is
not clipped. Source multiplicities and electron parity are checked before
training. Removing the legacy marker changes charged-system warm-start
behaviour even with a zero-initialized adapter; it is not a bitwise-preserving
reinterpretation of the old checkpoint.

The initial experiment is a matched 10,000-update continuation of the same
smoothed displacement checkpoint, with and without the global-state adapter.
Both use the same data order, FM-noise seeds, AdamW settings and prior width.
Dataset-side RNG is reset after construction so adapter initialization does
not change data-side randomness. This is an empirical OMol geometry FM baseline,
not energy-corrected or Boltzmann training. Requested kT=1 is a model condition
for future teacher training, not an asserted thermal label for the OMol corpus.
Checkpoints store the electronic protocol and prior width explicitly. A later
physics student must use a prior/target pair with appropriate tail coverage;
changing the prior of an old checkpoint alone is a separate counterfactual.

The metadata reader requires complete replay and verifies the processed-file
hash. Checkpoint loading reapplies the adapter before strict state_dict loading.
The conditional sampling panel attaches the verified electronic state and
uses the recorded prior width. xTB evaluation now uses supplied multiplicities;
minimum parity is a declared fallback only for older files without metadata.

The full suite passes 160 tests, including adapter gradients, state sensitivity,
missing-state and parity guards, removal of the atom-zero marker dependence,
hook cleanup and an independent-evaluator triplet regression. A two-update real
FlowMol preflight has finite weights, and its checkpoint reloads and generates
on held development compositions. No molecular-quality improvement is yet
established by this interface verification.

The first complete replay used SQLite WAL, which does not support live
cross-host readers on this filesystem. Read-only NumPy access was restricted
to the committed prefix while it ran. After clean completion the database was
exported to `source_index_readonly.sqlite` with DELETE journaling; integrity
and row counts passed. Readers should use `mode=ro&immutable=1` on that export.
Future replay runs use DELETE journaling directly. See `index_export.json`
and the raw-recovery/electronic-state evidence files for exact hashes.
