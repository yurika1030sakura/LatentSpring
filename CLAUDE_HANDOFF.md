# Claude continuation — September 12, 2026

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read `CLAUDE.md`, `research/NEXT.md`, `research/STATUS.md` and
`research/CONDITIONAL_NONLOCAL_STATE_20260912.json`. Do not write to the old home
checkout or merge the two Python environments.

The new scalar conditional energy is implemented, trained and audited. All four
fixed800-step models improve internal withheld nonlocal work prediction by about
19--24%, with no optimizer checkpoint selection. Full dataset and metric replay
pass. The separate `arc_energy` decoder integrates the actual scalar law into
complete joint graph/radius/two-root updates. It passes independent density and
public MH checks. Do not pass this model through the old vector `arc_model`.

All this round's jobs are terminal:46178737/46178918 and46179382/46180176.
There is no learned complete-chain result yet. NEXT specifies the48 internally
withheld-parent comparison, exact21,006-call method/replica accounting and66,588
maximum new raw calls. Implement per-parent physical caps before launching it.
Do not retrain completed models, restart completed jobs, or call prediction gains
an ICLR-ready sampling contribution. The prior physical site-arc benefit and all
negative controls remain recorded. Keep reserved722 outcomes untouched.

Prior handoff: `notes/archive/claude_handoff_through_geodesic_physics_20260912.md`.
