# Rendering from coordinates, then composing the figure

Use PyMOL for molecular shape, then a vector layout tool for labels and diagrams.
Here, chemistry's Python has PyMOL3.1; the FlowMol Python has working Matplotlib.
The chemistry environment's Matplotlib currently has a NumPy ABI conflict: do
not alter that shared environment to draw figures. Export numeric JSON between
the two environments instead.

The renderer consumes `{"scenes": [...]}`. Each scene needs `name`, `positions`
(N by3), `symbols`, and explicit `bonds` as `[i,j,order]`. An empty bond list means
no bonds are drawn. Optional `scaffold` edges are separate dashed auxiliary links.
Optional `camera_group` keeps related scenes on a shared camera and scale. The
positions should already contain the chosen rigid viewing rotation.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME=/tmp/lspring_figure_cache XDG_CONFIG_HOME=/tmp/lspring_figure_config \
QT_QPA_PLATFORM=offscreen \
/n/home04/yulili/.conda/envs/chemistry/bin/python \
  .agents/skills/scientific-figure-design/scripts/render_molecules.py \
  --scenes path/to/scenes.json --out path/to/renders --style outline
```

`soft` uses modest specular light and subtle occlusion. `outline` uses thin edges,
lighter carbon atoms, smaller hydrogens, and restrained element colors. Compare
the rendered result at its final page size; outlines that look refined at1200px
can disappear or become heavy when reduced.

Rendering choices should preserve:

- Every supplied atom, including hydrogens, unless a deliberate display omission
  is explicitly explained. Hiding atoms is not molecular relaxation.
- The supplied edge identities; do not let an XYZ reader infer new bonds.
- The input coordinates to PyMOL float32 precision. Record before/after error.
- A shared camera for any geometric comparison; do not independently orient
  before and after structures when displacement is the point of the figure.

Use near-tight framing with a modest margin. If trimming transparent pixels,
compute a union crop for all related frames so apparent scale does not change.
Keep text outside raster assets and preserve text as text in SVG/PDF. Never crop
or paint away a scientifically inconvenient atom, force arrow, or failed sample.

The numeric defaults in the script were informed by the
[PyMOL gallery](https://pymolwiki.org/Gallery) and
[ray-tracing documentation](https://pymol.sourceforge.net/newman/user/S0275ray.html),
then adapted to small molecules. They are starting points, not universal settings.

Native execution found two PyMOL details worth preserving: loading a model can
sort atoms by name, so compare coordinates by explicit atom identity rather than
array order; and shutdown hooks can mask an uncaught Python exception as exit0.
The helper restores identity order and explicitly returns nonzero on failure.
Its completed render manifest, not an empty stderr or successful launch alone,
records what was actually produced.
