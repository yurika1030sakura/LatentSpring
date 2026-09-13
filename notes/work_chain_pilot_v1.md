# Frozen complete-chain pilot for learned work policies

Protocol: `research/evidence/work_chain_protocol_v1.json`. This is a bounded
development experiment, not a stationarity certificate or final benchmark.
It uses the same18 audited starting states across six compositions, two frozen
model/sampling seeds and64 raw oracle calls per trajectory, including fresh
initialization. All216 trajectories and any failed cap must remain visible.
The maximum new physical cost is13,824 inversion-orientation calls.

| Arm | Chemical proposal | Additional learned-label calls |
|---|---|---:|
| root_noise | Existing uniform terminal exchange with small root-only noise | 0 |
| single_linear | Normalized valid-single-edit policy from typed bond work | 966 |
| panel_linear | Retained random cooperative panel, additive work only | 966 |
| panel_restraint | Same panel plus known confinement interaction preference | 966 |
| panel_radial | Same panel plus typed radial electronic interaction | 1,900 |
| panel_blind | Same panel plus context-blind neural electronic interaction | 1,900 |

These label counts are marginal calls for the existing work catalogues. They do
not include every historical training-start preparation cost. The common
evaluation-state preparation cost2,282 calls. No cold-start or total-cost win
may be inferred by omitting these distinctions. The contextual network remains
available in earlier results but is not expanded in this pilot.

All arms keep the existing repeating background schedule: local MALA,
force-informed terminal rotation, chemical move, local MALA. Only the chemical
move changes. Joint-step acceptance variates are drawn before candidate/panel
construction and keyed by composition, parent, replica and step. Different
candidate counts cannot shift another parent's stream or its acceptance draw.

The chemical adapter in `cfm_mol/work_chain_kernel.py` computes the actual
forward and reverse normalized catalogue probabilities. Cooperative selection
retains the same random panel at both endpoints. The interaction is a symmetric
preference, not a directional work correction. If no type-admissible four-root
block exists, cooperative arms use the same single-edit model. This predicate
is invariant under a terminal exchange, so it contributes no asymmetric family
factor. A sampled panel with no geometrically valid matching is instead a
self-loop; no geometry-dependent fallback is introduced.

Every scored rejection costs its two raw orientations. Invalid geometry costs
no energy call, but attempts and runtime remain recorded. Each completed arm
publishes a partial trace and charged-query progress. Failures preserve available
states and raw query responses. Source snapshots and historical runs are not
overwritten. Model and source hashes are checked before execution.

The two primary observations, fixed before outcomes, are the number of distinct
stereo-stripped constitutional connectivities visited and the actual potential
change at64 calls. Readouts also exist at16 and32 calls. Secondary observations
include connectivity transitions/returns, accepted uphill moves above kT, and
typed sorted pair-distance movement. The latter excludes pure rotations and
same-element relabelling but is not a complete geometric invariant. None of
these quantities identifies a thermodynamic basin or certifies mixing.

A query-budget stopping rule depends on proposal validity. Even though each
individual kernel has the correct MH correction, its budget-stopped endpoint
is not thereby an equilibrium sample. A finite-temperature exploration
hypothesis must not silently become a claim of matching the target energy
distribution, particularly if energy is still drifting on these short chains.

The independent audit reconstructs raw target potentials, clipped-MALA and
rotation MH ratios, physical root-noise inverse maps, stable scalar catalogue
normalizers, selected-action MH ratios, every accepted-state history update and
every query count. The new adapter and the unchanged legacy driver both pass
mixed-kernel replay tests. All readout summaries are replayed from the traces.
The final summarizer reports all15 pairwise method comparisons, not only a
favorable neural control, and descriptive parent intervals within each fixed
composition. Both replicas stay together in resampling. No multiplicity or
untouched-test guarantee is implied.

An overall advantage requires comparison with the strong single-edit policy and
cheap physical moves, not merely improvement inside the cooperative family.
Small exploratory differences accompanied by substantial runtime overhead do
not justify scaling the current recipe or promising ICLR readiness.
