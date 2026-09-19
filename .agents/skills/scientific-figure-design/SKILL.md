---
name: scientific-figure-design
description: "Design, render, and visually inspect publication figures: molecular illustrations, method diagrams, and quantitative panels. Use for figure aesthetics or readability work, with actual geometry and data retained."
---

# Scientific Figure Design

Treat scientific content and visual craft as separate review dimensions. A figure
can be factually correct and still look poor. A successful renderer invocation
or a clean PDF build does not establish visual quality.

## Start with the final viewing size

Read the manuscript's actual `textwidth` and planned figure width. Design at that
physical size, or calculate `effective_font_pt = canvas_font_pt * placed_width /
canvas_width`. For this project's ICLR template, full width is **5.5 inches**.
The earlier 7.5-inch canvases reduced 6.5-point labels below 5 points; avoid that
failure. Use about 8-point labels and 10-point panel titles at placed size as a
starting point, then inspect. These are local working choices, not venue rules.

## Make deliberate visual choices

- Look at relevant, well-designed published figures or primary design tutorials.
  Record the specific technique to borrow, not just a prestigious paper name.
  Read [references/design-sources.md](references/design-sources.md) for sources.
- Decide what first attracts the eye, what should be compared, and where reading
  proceeds next. Give the important scientific object more area than its labels.
- For schematics, illustrate states and label the transformations between them.
  Keep an object recognizable as it changes. Avoid repeating near-identical
  snapshots unless they reveal a useful change.
- Choose a clear visual hierarchy: a focal object, supporting mechanism, then
  annotations. Large uniform tinted cards and repeated bold headings are poor
  defaults for this project's molecular figures; use white space and alignment.
- Use a compact semantic palette. Atom colors encode elements; process accents
  encode operations. Do not use saturated accents for all objects simultaneously.
- Match arrow width, corner radius, and line weight across a diagram. Route
  arrows around labels. Borderless grouping is often enough; boxes should signify
  an actual module, boundary, or container.

## Molecular illustrations

Read [references/molecular-rendering.md](references/molecular-rendering.md).
Use `scripts/render_molecules.py` with numeric scene JSON for reproducible PyMOL
renders. It offers `soft` and `outline` starting styles. Compare both on the same
coordinates and camera before choosing; a preset is not a quality guarantee.

Reduce atom occlusion with a rigid viewing rotation, choose readable sphere/stick
ratios, and use lighting that shows shape without distracting glare. Crop empty
canvas, not atoms. Paired structures and trajectory frames share camera and scale.
For unrelated gallery examples, adjust framing only and retain the sample rule.

Never move atoms, add inferred bonds to a source cloud, smooth a trajectory, or
optimize coordinates to improve an evidence figure. Preserve atom counts, explicit
bond provenance, sample IDs, and geometry hashes. Any schematic exaggeration must
be labeled and kept distinct from measured geometry.

## Quantitative panels

Use vector axes, lines, and live text. Plot the recorded values and preserve the
statistical unit of intervals. Prefer direct labels when they reduce eye travel.
Remove unnecessary grid lines, redundant legends, and repeated captions inside
the plot. Do not truncate axes or hide runs to make an effect look larger.

## Visual review and delivery

Render and inspect the actual files, not just the plotting code. Check both the
standalone figure and the compiled paper page at normal reading size. Compare
against the previous version using the same display scale.

Look specifically for tiny molecules, hidden atoms, plastic-looking highlights,
weak contrast, crowded text, arrows touching equations, misaligned baselines,
and surplus empty borders. Also inspect a grayscale preview. These checks guide
judgment; they are not a mechanical aesthetic score.

Retain editable SVG/PDF text and a high-resolution preview, the script and source
hashes, and a short record of visual decisions. If seeking aesthetic feedback,
show the figure itself. Do not describe a design as beautiful or accepted on the
user's behalf. Existing authorization governs publishing; this skill adds no
permission workflow.
