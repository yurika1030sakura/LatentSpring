# Figure regeneration

Run in order with dependencies from `requirements.txt`:

```bash
python refresh_plot_data.py
python build_figures.py
python normalize_figure_fonts.py
python export_assets.py
python audit_figures.py
```

The refresh recomputes Figure 2 from the supplied per-attempt NPZ, CSV and JSONL
records. The build renders Figures 1 and 2. Other figures are rebuilt from the
supplied font-normalized layouts in `normalized_svg/`, with glyph definitions
from `original_svg/`. Curves, points, intervals and molecular coordinates stay
unchanged. The normalizer does not require Inkscape.

PNG and `Figures.pdf` exports derive from the final PDFs. Outlined SVG export
uses Inkscape or `pdftocairo` (Poppler). Editable SVGs stay in `../figures/`.
Scripts work without a manuscript directory; PDF synchronization runs only if
`../manuscript/figures/` already exists. Audits are written under `audit/`.

No checkpoint, generation, energy query or geometry optimization is required.
