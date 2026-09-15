# LatentSpring submission draft

Canonical manuscript: `paper/tree_working.tex`; compiled PDF: `paper/latentspring.pdf`.
The default `main.tex` and legacy `bgfm_paper.tex` entrypoints use this manuscript.
The self-contained `paper/latentspring_overleaf.zip` builds independently from `main.tex`.
Submission title and plain-text abstract: `paper/submission_abstract.txt`.
The working draft targets ICLR2027. It is not a published or accepted paper.

The paper focuses on its specified molecular source, physical-feedback update and
controlled work correction. It does not claim unrestricted Boltzmann generation,
rapid optimization to quantum minima, or native published-SOTA superiority.
Discarded exploratory experiments remain in the project record and are not part
of the method being submitted.

## Evidence supporting the reported claims

- Source comparisons: `research/evidence/source_sc_audit_v1.json`,
  `source_sc_confirmation_audit_v1.json`, and `source_sc_covariance_audit_v1.json`.
- Fresh24 structural and physical confirmation: `fresh_physics_esen_audit_v1.json`
  and `fresh_physics_xtb_audit_v1.json` with their hashed metric arrays.
- Independent adapted EDM: `conditional_edm_extended_audit_v1.json` and its arrays.
  This uses30000 continuation updates; backbones and pretraining histories differ.
- Controlled molecular work: `rotor_work_v1.json`, `rotor_work_results_v1.json`,
  `rotor_work_energies_v1.csv`, and `rotor_work_audit_v1.json`.
- Current campaign ledger: `generator_reference_registry_v13.json`.
  It distinguishes model outputs, numerical draws, diagnostic copies and oracle calls.

All listed evidence files are under `research/evidence/`. The rotor configuration,
energy table and full statistical summaries are small version-controlled files.
The complete raw job artifacts remain under `runs/`, with source/checkpoint/input
hashes preserved by the audits. Model and dataset access follows their original
licenses; no credentials are embedded in this document.

## Rebuild the work figure

From the project root, with the project Python environment:

```bash
python scripts/research/plot_rotor_work.py   --results research/evidence/rotor_work_results_v1.json   --audit research/evidence/rotor_work_audit_v1.json   --out research/figures/rotor_work_v1
```

The figure averages all three molecules and both nonidentity escorts. The identity
control and every prescribed sample size remain in the full result file.

## Compile the manuscript

From `paper/`, run pdflatex, bibtex, and pdflatex twice on `tree_working`.
The final build evidence records all recursively used section and figure hashes
in `research/evidence/publication_build_v1.json`.

## Training scope

OMol25 is the primary data source. The source comparisons use pretrained backbones
and3000 selected training structures per continuation. This is not full-corpus
OMol25 training. Core source controls share their data, backbone, initialization,
optimizer and inference budget. The separately adapted EDM has a larger continuation
budget and different pretraining history, explicitly reported in the paper.
A full-corpus run is not required to reproduce or support these scoped comparisons.
