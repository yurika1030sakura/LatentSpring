# Next — finish and audit the frozen source utility pilot

CURRENT OVERRIDE (2026-09-14): source-utility method IMPLEMENTED; controlled jobs
46434075_0/1 RUNNING from immutable commit0ad2193. Read
`research/SOURCE_UTILITY_STATE_20260914.json`, `notes/source_utility_method_v1.md`
and `research/evidence/source_utility_protocol_v1.json`. Prior “not implemented”
statements below describe the completed monomer checkpoint, not current work.
There is no new validated utility gain yet.24 FIT plus12 source-head held-out
compositions come from the real TRAINING corpus; do not confuse this split with
the older21 composition-disjoint evaluation molecules.9216 planned attempts,
actual/shuffled/NLL/fixed controls, two fixed decoders, no oracle calls. Re-query
Slurm; finish and audit this bounded run before proposing another method. Generic
frozen-decoder noise learning is prior art (Noise PPO); originality remains open.


Read `research/MONOMER_STATE_20260914.json`, `notes/monomer_results_20260914.md`
and `notes/source_utility_learning_brief_v1.md`.

The task/readout mismatch has been addressed prospectively for21 neutral singlet
organic monomers8-40 atoms. All references passed before generation; exact equal
compositions are absent from two checksum-verified processed corpora. Full arbitrary
pretraining/trajectory disjointness is not claimed. All10,752 new outputs replay.

Graph passes/1,344: first Gaussian/shell/harmonic/node/pair581/627/575/603/561;
second Gaussian/shell/harmonic521/557/598. Node/pair are one-continuation evidence
only and do not beat fixed shell; pair is worse under both uncertainty summaries.
Shell geometry improves, but its paired graph intervals span zero. Keep both
conditional-draw and descriptive-composition intervals; do not choose the more
favorable one or a different source by seed. All earlier negative studies remain.

Next authorized bounded work:

1. Assess the single new hypothesis in the source-utility brief: keep the actual
   generator kernel frozen and learn the normalized source from terminal utility.
   This tests objective alignment without simultaneously adding a context layer,
   enlarging the backbone or changing source kernels.
2. Check prior work on latent/noise adaptation, reward-weighted fitting and
   generator steering. Importance identities and data processing are not our
   inventions. Establish the exact source objective, support assumptions, trust
   control and estimator behavior before a performance claim.
3. Select disjoint FIT compositions from the real training corpus under the
   coherent monomer criteria, then freeze a bounded source/output bank, training
   budget and fresh validation streams. No prior evaluated coordinates or outcomes
   may enter that bank. Count preparation and validation generation costs.
4. Compare actual utility, within-composition shuffled utility and unadapted source
   with identical head capacity and fixed decoder. To attribute a change specifically
   to the objective, include a corresponding coordinate-NLL source under that same
   decoder. Prespecify trust limits and stop criteria; no utility-trained model or
   protocol is implemented/frozen yet. Offline importance gains alone are insufficient.

Do not merely scale the old source-NLL heads or static context adapter on the
observed panels. Their failure does not uniquely diagnose the cause, and the new
hypothesis may also fail. A useful, sufficiently distinctive method remains the
ICLR objective, not more modules or a nearly perfect generator on every element.

Current artifacts:

- `monomer_panel_v1.json`, `monomer_qualification_v1.json`:prospective selection
  and all78 reference decisions under `research/evidence/`.
- `monomer_overlap_v1.json`, `monomer_overlap_controls_v1.json`:actual-corpus exact
  composition checks and positive controls.
- `monomer_evaluation_audit_v1.json`:all sources/outcomes, matched training rows,
  checkpoint identities and uncertainty summaries.
- `research/figures/monomer_v1/`:complete frozen-model figure.
- `notes/ai_novelty_monomer_scope_v1.md`:candidate contribution and prior-art boundary.
- `paper/tree_working.tex` and `.pdf`:internal working manuscript, not submission ready.

Jobs46415396_0/1 completed0:0 in34:26/20:28; query live Slurm.17 targeted source/
restoration tests pass. No new fitting or physical queries in this evaluation.
The earlier tree-energy readout remains5,120 calls. Audits do not independently
retrain optimizers or regenerate the final neural integration.

Protect722 reserved outcomes, old evaluated cohorts,2,560 orbit outputs,30,208
tree/monomer outputs,32 old references and78 monomer-pool references from fitting.
Keep bond-free OMol25, max_atoms200, original charge/spin, separate environments
and no home writes. No final density/ESS/Boltzmann claim. Old failed routing,
collision, latent-mass and static-context sweeps remain closed. User handles
submission; the ICLR goal remains active and unachieved.
