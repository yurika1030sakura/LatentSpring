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
- Current campaign ledger: `generator_reference_registry_v15.json`.
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
in the latest `research/evidence/publication_build_v3.json`.

## Training scope

OMol25 is the primary data source. The source comparisons use pretrained backbones
and3000 selected training structures per continuation. This is not full-corpus
OMol25 training. Core source controls share their data, backbone, initialization,
optimizer and inference budget. The separately adapted EDM has a larger continuation
budget and different pretraining history, explicitly reported in the paper.
A full-corpus run is not required to reproduce or support these scoped comparisons.

## Expanded evidence and component tests

- `wide_generalization_audit_v1.json`: frozen models on64 further compositions.
- `matched_generators_audit_v1.json`: independent EGNN source and diffusion study,
  with20,000 shared training rows and two unpretrained initializations.
- `component_benefits_audit_v1.json`: SC controls with equal update/forward budgets
  and physical-update direction versus a norm-matched direct update.
- `force_shift_ablation_audit_v1.json`: the force displacement versus identical
  local candidate draws before displacement.

These audits are under `research/evidence/`. Run files and weights from the
expanded campaign have been copied to permanent HOLY storage with every file
verified byte-for-byte; their `runs/` entrypoints are retained. The archival
manifest is `research/evidence/retained_benchmarks_v1.json`. Original scratch
copies remain available. The standalone paper package includes its full notation
guide and derivations; it does not redistribute the underlying licensed datasets.

## Latest physical readouts and work-sampling tests

- `generator_quality_audit_v1.json`: unchanged matched-generator outputs scored
  with eSEN and GFN2, with all-attempt physical-quality yields.
- `curvature_escort_audit_v1.json` and `curvature_full_teacher_audit_v1.json`:
  independent density checks and local ESS comparisons at matched maximum query
  budgets. Curvature improves the teacher ESS.
- `curvature_distillation_esen_audit_v1.json` and
  `curvature_distillation_xtb_audit_v1.json`: paired neural students and
  fixed-coordinate evaluation. The additional neural gain from curvature work
  was not established; the primary model retains the force-based update.

## Molecular figures and supplementary animations

The main figures use original coordinate records, with element colors and rigid
viewing rotations. They do not use generated artwork as molecular evidence.
`molecular_overview_v2/provenance.json`, `molecular_rotor_v2/provenance.json` and
`raw_molecular_gallery_v1/provenance.json` record their data sources and display
rules under `research/figures/`.

From the project root in the FlowMol environment, regenerate them with:

```bash
python -m scripts.research.make_molecular_overview
python -m scripts.research.make_molecular_rotor_figure
python -m scripts.research.make_molecular_gallery
```

The recorded trajectory is under `runs/editorial_review_20260915/trajectory/`.
It replays 16 existing outputs, with maximum coordinate disagreement below
5e-6 angstrom, and adds no benchmark outputs or energy queries. Its animation
shows flow time. The rotor animation shows the prescribed torsional coordinate
used in the controlled work experiment. The source package includes both GIFs
as supplementary files, alongside static PDF figures.
