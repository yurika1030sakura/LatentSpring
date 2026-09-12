# Current research status — September 12, 2026

Current checkpoint: `research/CONDITIONAL_NONLOCAL_STATE_20260912.json`.
The ICLR goal remains active and scientifically unachieved. Read
`research/CLAIM_AND_BENCHMARK_SCOPE.md` for the user's intended standard.

The new scalar conditional energy model is implemented, trained and independently
checked. It predicts context-conditioned root-to-passive interaction curves,
learns potential differences and optional tangential forces, and supplies an
exactly evaluated interpolated-circle proposal through `arc_energy`. All actual
partial contexts, both decoder orders, radius factors and circle preimages enter
the forward/reverse density. The old one-vector `arc_model` has separate semantics.

Training data contain438 contexts:206 FIT,76 withheld-parent and156 withheld-
composition. FIT local-check directions and all withheld labels stay excluded
from fitting. Four fixed800-step trainings compare work-only versus work-plus-
force, with two seeds each and no early checkpoint selection. No new physical
calls are needed for fitting. The928 extra withheld arc endpoints cost1,856 raw
calls; their full independent replay is complete.

All four final models improve internal nonlocal work prediction over the physical
site64 score. Withheld-parent MAE falls from0.37036 eV to0.28675--0.29839 eV;
withheld-composition MAE falls from0.58831 eV to0.44725--0.47630 eV. Tangential
force errors also improve. These are conditional prediction diagnostics, not
complete-chain efficiency, blind final evaluation or established AI novelty.
The force-loss addition has no demonstrated decisive benefit over work-only.

The learning audit rebuilds all438 contexts and split masks, reloads every model,
reproduces every final context metric and both physical baselines, and checks
actual trained-loss finite differences (maximum error1.19e-8). The optimizer's
full trajectory has not been replayed. Source and model hashes are frozen.
Audit: `runs/conditional_arc_learning_audit_v1/results.json`.

| Completed work | Producer | Independent audit |
|---|---|---|
| Internal withheld arc labels | 46178737 | 46178918 |
| Four scalar-energy trainings | 46179382 | 46180176 |
| Complete physical site/uniform arc chains | 46176878 | 46176906 |
| Real complete joint support | 46174307 | 46176033 |

All these jobs are terminal. The new scalar model has no learned molecular
complete-chain experiment yet. NEXT specifies the next cost-matched pilot.

The preceding physical arc result remains useful: on48 FIT parents, site arcs
improve the128-call potential-change readout by0.07615 eV against legacy site64,
with22.4% more sampling runtime. Uniform arcs show no established average chain
benefit. This does not overturn the previous learner's failure against strong
concentration64/400 controls or establish transfer of that old learner.

All training-source support counts remain45,32,106,99,0,19,0,96 out of256 each.
Both zeros are retained; condition6 has empty connected support under the pinned
builder's necessary degree bound, while condition4 remains unresolved. Do not
replace them. Original charge/spin, all failed attempts and negative controls stay
in the evidence. The722 reserved outcomes remain untouched.

Raw cost accounting for the next internally withheld-parent pilot:20,950 total
preparation/probe/label calls, of which6,088 are preparation shared with its48
starts. The model-specific overhead is14,862 calls. Count shared preparation
once; record the previous36,864-call physical route pilot separately. FM/source
and fitting/runtime costs remain explicit in the final benchmark.

The latest reviewed PDF is
`runs/verification/conditional_nonlocal_20260912/main.pdf`, eight main pages
and18 total. It contains both the physical reconstruction and the new scalar
learning diagnostics, with prior-art citations, cost accounting and limitations.
The new table is visually checked; page, citation and layout checks pass. Prior status is in
`notes/archive/status_through_geodesic_physics_20260912.md`.
