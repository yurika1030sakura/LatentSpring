# Scope of the endpoint-correction contribution

Primary sources checked on 2026-09-20:

- Corso et al., *Composing Unbalanced Flows for Flexible Docking and Relaxation*,
  ICLR2025, Section4.2/page8. FLEXDOCK evaluates a geometric energy loss on late
  predicted endpoints. Its pair-distance bounds use chemical connectivity and
  its inference includes energy/confidence filtering.
  https://proceedings.iclr.cc/paper_files/paper/2025/file/451dbb8f4fca0327ac4e6782786673bf-Paper-Conference.pdf
- Wang et al., *Protein Conformation Generation via Force-Guided SE(3) Diffusion
  Models*, ICML2024. ConfDiff includes a learned force-guidance network for
  protein conformations. Learned physical guidance is established prior work.
  https://proceedings.mlr.press/v235/wang24cv.html
- Lai et al., *Improving protein-ligand complex generation with force field
  guidance*, Journal of Cheminformatics18:55,2026. The method evaluates MMFF94
  during diffusion/flow inference, using molecular graph attributes.
  https://doi.org/10.1186/s13321-026-01198-2

The manuscript now cites all three. Do not claim the first predicted-endpoint
energy objective, first learned force network, or first physical correction of
both flow and diffusion generators. The velocity/noise conversion is an
algebraic endpoint identity, not a new physical law.

The concrete contribution to defend is the composition-conditioned method:
a normalized auxiliary-tree source without supplied chemical bonds, combined
with a bounded equivariant head trained from bond-free forces at generated
provisional endpoints, keeping the parent frozen and avoiding online oracle
calls. Its value requires controlled source and correction evidence, including
equal supervision for the competing generator.

`seed_replication_v1.json` freezes three new fit pairs and64 new compositions;
its primary independent-training analysis uses the three new fits. Separately,
`cross_generator_head_v1.json` freezes verbatim head transfer between FM and
GAGA, without retraining or target-parent force labels for the transferred head.
If it succeeds, this supports weight reuse across those two objectives on the
shared architecture. It does not automatically establish transfer to every
architecture, chemistry, or energy potential. The cross-head results are not
qualified while jobs are running.
