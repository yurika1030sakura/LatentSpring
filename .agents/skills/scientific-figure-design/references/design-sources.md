# Primary sources and the decisions they inform

Consulted online on 2026-09-19. The principles below are paraphrases; the
working dimensions and styling defaults are project choices, not requirements
imposed by these sources. Reference artwork is for study, not for republication.

| Source | Practical use |
|---|---|
| [Nature research figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/) | Check final dimensions, editable text, resolution, accessible color, and export format. Apply the target venue's dimensions rather than copying Nature's sizes. |
| [MIT Communication Lab: Figure Design](https://mitcommlab.mit.edu/nse/commkit/figure-design/) | Establish the visual message first, choose the representation accordingly, reduce competing decoration, and label the relevant object directly. |
| [Wong: The overview figure](https://www.nature.com/articles/nmeth0511-365) | Show the objects before and after a meaningful operation; preserve recognizable visual correspondence between stages. Only the accessible opening discussion was used. |
| [Wong: Typography](https://www.nature.com/articles/nmeth0411-277) | Treat typeface, spacing, size, and hierarchy as part of scientific communication. The accessible text and figure descriptions were consulted. |
| [Rougier et al.: Ten Simple Rules for Better Figures](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003833) | Customize software defaults and design for the actual publication medium. A good caption supports the figure; it does not rescue unreadable rendering. |
| [Mura et al.: An Introduction to Biomolecular Graphics](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000918) | Choose representation and view for the structural feature; use scripted scenes, appropriate lighting, and separate composition tools when useful. |
| [PyMOL: Render Figures Artistically](https://www.pymol.org/render.html) | Ray tracing improves edges and shading, but camera and representation still determine what the reader sees. |
| [PyMOL ray-tracing manual](https://pymol.sourceforge.net/newman/user/S0275ray.html) | Orthographic projection, ambient/direct illumination, antialiasing, and specular settings are explicit controls. The manual is old; verify settings against the installed version. |
| [PyMOL community gallery](https://pymolwiki.org/Gallery) | Study the stylized ball-and-stick and Goodsell-like examples. Compare restrained outlines with soft shading using our own structures. |

## Project-specific visual lessons

The earlier publication used small molecular renders inside large tinted cards,
long labels, and a canvas wider than the final page placement. The result was
readable only after zooming and looked like a slide template. The correction is
to design at 5.5-inch width, enlarge the molecular subjects, use shorter live-text
labels, and reserve color and enclosure for meaningful mechanisms.

Previously inspected ConfGF and Pocket2Mol figures illustrate concrete atom-level
operations and compact state changes. Their molecular task inputs differ from
LatentSpring: do not copy known-bond or protein-pocket inputs into our diagram.
Local reference images and the earlier review are documented in
`notes/luo_shi_writing_and_figures_20260915.md` at the project root.
