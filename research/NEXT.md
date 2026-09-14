# Next — align the single-molecule task and assay before more architecture variants

Read `research/TREE_TRANSFER_STATE_20260914.json` and
`notes/tree_results_and_scope_20260914.md`. The ICLR goal is active and unachieved.
The user wants a useful AI contribution and a fast real demonstration, not universal
perfection. The current evidence does not establish ICLR submission readiness.

Completed evidence:

- Static tree context: no-context / actual / independent graph passes/512 are
  118/102/119 and97/75/97. Actual context has no useful increment; second-seed
  negative intervals exclude zero. Close this adapter recipe to scale-up/retuning.
- 32-composition source transfer: Gaussian / shell / harmonic graph counts/1,024
  are180/200/183 and164/178/195. Shell geometry gains repeat, while graph-gain
  intervals span zero. Do not select a different source separately by seed.
- Reference assay: only6/32 original references pass graph perception;20 pass
  geometry,12 are disconnected and12 cause validator errors. All16 all-arm-zero
  conditions also have rejected reference graphs. Raw references may be reactive,
  distorted or multicomponent; this alone does not identify a validator bug.

Next authorized work:

1. Define a coherent primary single-molecule generation domain and audit its
   reference-assay coverage before querying new generated outputs. A connected
   organic-monomer domain with supported electronic conditions is a possible
   scope; state its limitations openly. OMol25 remains primary and max_atoms200.
2. Inspect the eligible source pool using original reference geometry and metadata,
   excluding declared earlier evaluation/training panels. Do not use generated
   performance or energy ranking to choose cases. Resolve pretraining overlap
   before making an unseen-composition claim. No replacement panel is frozen yet.
3. Keep the current broad molecular-system panel as a separate stress test with
   geometry/energy/assay-coverage reporting. Do not replace frozen primary counts
   by a post hoc reference-passing subset or reinterpret rejected outputs as valid.
4. Once the task/readout is coherent, test frozen source baselines and then one
   mechanism-based learned improvement. Another attention layer, lower prior NLL,
   or lower FM loss is not a demonstrated molecular gain. No new physics law and
   no success on every element are mandatory; the actual claim needs evidence.

Artifacts:

- `research/evidence/tree_context_audit_v1.json`: all3,072 context-study outputs,
  exact common sources, actual/sham trees, matched learning budgets and restoration.
- `research/evidence/tree_source_transfer_audit_v1.json`: all6,144 transfer outputs,
  source/checkpoint identities, per-case outcomes and two uncertainty summaries.
- `research/evidence/tree_reference_assay_v1.json`: original-reference calibration;
  coordinates in `tree_panel_reference_coordinates_v1.json` are evaluation only.
- `research/figures/tree_context_v1/` and `tree_source_transfer_v1/`:PDF/PNG figures.
- `paper/tree_working.tex` and `.pdf`:internal manuscript preserving both gains
  and negative findings. `paper/tree_abstract.txt` follows its current evidence.
- `notes/tree_context_related_work_v1.md`:latent/mixture conditioning is prior art;
  neither a residual adapter nor a tree theorem certifies originality.

Four scheduled tasks46367714_0/1 and46368764_0/1 completed0:0; query Slurm before
recovery.57 targeted tests pass. These two studies use zero new oracle calls;
the earlier physical readout remains5,120. Audits do not independently retrain
optimizers or regenerate final neural trajectories. No final density/ESS/thermal
claim is licensed. The static adapter is retained for reproducibility and its
conditional-density guard must not be removed.

Protected:722 reserved outcomes; old12/18 cohorts;2,560 orbit outputs;19,456 tree
outputs;32 raw reference geometries. None may enter fitting. Preserve charge/spin,
bond loss zero, separate environments and no home writes. Earlier failed routing,
collision and latent-mass sweeps stay closed. User handles actual submission.
