# Claude continuation — work prediction for chemical edit choice

Active checkout: `/n/holylabs/ryl_lab/Lab/yulili_cfm_mol/iclr2027`, branch
`iclr2027-arch-fix`. Read current NEXT and
`research/CHEMICAL_WORK_POLICY_STATE_20260913.json`, then
`notes/chemical_work_policy_candidate_v1.md`. Never write home or merge environments.
The user prioritizes AI framework and fast evidence, not perfect physical cases.

The learned-mobility route is not a demonstrated AI contribution. Its earlier
arc-baseline advantage disappears against supplementary simple controls:
collective -0.60935 eV, bare -0.59773, small root noise -0.61205 at128 calls.
No repeatable learned advantage over these controls is established. The full
18-parent/seeds/source/query contracts and every result remain in
`research/evidence/edit_mobility_strong_control_comparison_v1.json`.
Do not scale/tune the old mobility weights or revive optimizer cleanup.

The next hypothesis learns finite chemical work to SELECT useful edits before
physical queries. Most bare proposals are geometrically valid but energetically
unfavorable. A candidate shared masked-context paired-energy decoder can encode
3D environments and guarantee paired reversal sign. It is not implemented yet.
Informed/locally balanced proposals, LSB and MOSAiCS are prior art; a generic
balancing rule or GNN is not sufficient novelty. See the design note for controls.

Data is READY: all492 eligible bare edits on36 original FIT sources;447 valid,
45 failed,966 raw physical calls. All four producers and audits are COMPLETE:
46227985/46228322. Independent physical-work reconstruction also passes.
26/36 sources have a downhill candidate; source-force linear work MAE is3.185 eV.
These are training opportunities, not an oracle sampler or method success.
Plan/data: `runs/chemical_work_catalogue_plan_v1/plan.pt` and
`runs/chemical_work_catalogue_v1/condition_*/trace.pt`.
Audit: `runs/chemical_work_catalogue_audit_v1`.
Summary: `research/evidence/chemical_work_catalogue_summary_v1.json`.

Next implement/train the compact representation with parent-balanced losses and
linear/graph-only controls, then a corrected action-selection molecular pilot.
Compute actual forward AND reverse normalized catalogue probabilities and retain
all invalid actions. Do not use candidate true energies in forward policy
inference. Keep all evaluated12/18-parent cohorts and722 reserved outcomes out
of fitting. All current jobs are terminal; re-query Slurm before recovery.
Goal active, scientific submission readiness false. Authors/submission stay with
user. Historical mobility details remain in `research/EDIT_MOBILITY_STATE_20260913.json`.
