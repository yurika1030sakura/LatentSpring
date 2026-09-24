# Figure 1 molecular rendering study

The copied `scenes.json` is byte-identical to the verified C7H15N scene in
`research/figures/geometric_method_v1`. The generated coordinates, explicit atom
identities and Kekule bond orders are unchanged. Source edges remain dashed
virtual springs. No bonds are inferred by PyMOL, no geometry is optimized, and
all 23 atoms are retained.

Five render styles were compared at 1800 by 1800 pixels, on the exact same camera
and scale: `outline`, `soft`, `polished`, `sculpted`, and `balanced`. I recommend
`balanced`: moderate diffuse shading, slightly larger spheres than the original,
minimal specular light, neutral gray carbon, blue nitrogen and visible white
hydrogens. `polished` looked too flat; `sculpted` looked too glossy at small size.

Use `balanced/source_paired_crop.png` and `balanced/output_paired_crop.png` with
the SAME placed width. The union crop is identical for all styles and scenes and
retains every nontransparent pixel, with a 32-pixel margin. The full-resolution
uncropped originals and render receipts are beside these files.

`style_comparison.png` is an internal visual contact sheet, not a paper figure.
`render_molecules.py` is an isolated copy of the project PyMOL renderer with only
rendering-style additions. `compose_study.py` makes the union crops/contact sheet.
`study_manifest.json`, `crop_manifest.json`, and each style's render manifest
record provenance and integrity checks.
