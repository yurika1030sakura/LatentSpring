# Experimental graphics and narrative alignment

The paper now follows the user's four-part experiment structure: setup, the
complete FM versus three baselines, a four-setting FM component study, and the
merged transfer/repetition section. Quantitative figures replace the three main
tables. Existing repetition, weight-transfer, and hydrogen evidence stays in the
main text. Pending combined-design outcomes have no data marks.

## Corrections to the scientific description

| Previous mismatch | Corrected description and supporting evidence |
|---|---|
| Abstract emphasized an older source study and separately enhanced GAGA before the current main comparison. | Title, abstract, introduction and discussion now foreground harmonic sources plus physical correction, the four-method comparison, and the independent one-pass component study. Numbers come from the completed replication and factorial audits. |
| Main self-conditioning description presented FlowMol's coordinate-shaped output as the default. | The main model is the shared EGNN velocity predictor. FlowMol's output interpretation remains documented with the earlier source experiments. See `cfm_mol/matched_egnn.py::vector`. |
| Gradient flow through the provisional endpoint was ambiguous. | Both passes are supervised, but distance feedback uses detached coordinates, matching `cfm_mol/connectivity_feedback.py::GeometryContext` and `prediction`. |
| Main sampling paragraph described the earlier FlowMol midpoint convention and terminal0.025A noise. | Main EGNN sampling uses current time followed by midpoint time,128 backbone calls plus64 head calls, with no terminal noise. The earlier FlowMol convention is preserved in its appendix. |
| Source tree was described as discarded after initialization without distinguishing the new diffusion extension. | This is now explicitly the FM rule. Combined-design diffusion keeps the sampled covariance fixed, conditions its denoiser on pair variances, and uses colored noise in training and reverse sampling. `harmonic_diffusion.py` and the frozen protocol specify the implementation. |
| Combined-design adaptation could be read as wholly zero-shot. | Text states that structured diffusion backbones are trained for their noise process; only physical-head weights are transferred without fitting. No improvement claim is assigned to pending outcomes. |
| The four-setting ablation could be confused with the final self-conditioned model. | Main text, caption, abstract and appendix identify it as the separate one-pass FM configuration sharing one frozen head per fit. |

Jarzynski/local-work and paired-weight updates remain complementary analyses,
with no global equilibrium-generation claim. The optional H flow remains
separate from the two-design main comparison and ablation.

## Figure decisions

- Main comparison: horizontal bars from zero, direct mean labels, and every
  individual fitted model shown as an open point. Two panels retain the two
  original metrics and method ordering.
- Component study: paired lines connect correction-off/on states; color and
  marker shape distinguish the source. Means and both fitted pairs are visible.
- Transfer: separate panels for the pending combined-design experiment and the
  completed five-fit physical-head/H study. Pending positions are not plotted
  as zero and do not receive invented connecting lines.
- Final figure width5.5in; editable PDF/SVG text; PNG and grayscale previews.
  Main text uses the experimental_story_v4 exports. Earlier visual iterations
  remain in the working runs for comparison.

Design references checked online: [MIT Figure Design](https://mitcommlab.mit.edu/nse/commkit/figure-design/)
informed direct labeling and choosing a visual for each experimental question;
[Nature figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)
informed physical sizing, editable output and readable text. No reference artwork
was copied. The protocol and completed audits remain the source of numerical values.
