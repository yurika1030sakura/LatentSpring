# Figure design research + caption drafts (BGFM / HBFM, ICLR 2027)

Written 2026-08-03. Companion to `paper/figures/fig1_method.tex`,
`paper/figures/make_experiment_figures.py`, `paper/figures/dump_boltz_records.py`.

Everything below is either (a) something observed in a specific published paper
(cited), or (b) a decision we made, with the reason. Nothing here changes a
result; where a figure shows a number that is not yet real it is drawn in red
and listed in the placeholder inventory at the end.

---

## 1. What Figure 1 looks like in this subfield

I read the figure captions / HTML of FlowMol3, EDM, Adjoint Sampling,
Transferable Boltzmann Generators, Sequential Boltzmann Generators and EnFlow.
Two distinct traditions exist, and they are used for different claims.

### Tradition A — "the pipeline strip" (one row, left to right)

Used when the contribution is *the generative process itself*.

* **EDM** (arXiv:2203.17003), Fig. 1: *"Overview of the EDM. To generate a
  molecule, a normal distributed set of points is denoised into a molecule
  consisting of atom coordinates x in 3D and atom types h. As the model is
  rotation equivariant, the likelihood is preserved when a molecule is rotated
  by R."* — literally a strip: Gaussian point cloud on the left, progressively
  structured molecule on the right, plus a rotation glyph to carry the
  equivariance claim.
* **FlowMol3** (arXiv:2508.12629), Fig. 1: single-column strip, continuous
  coordinates via ODE and discrete atom types/charges/bonds via CTMC drawn on
  the same time axis; the three contributions (self-conditioning, fake atoms,
  geometry distortion) are called out as annotations on the strip rather than as
  separate panels. Their Fig. 2 is then a pure block diagram of the network.
* **Adjoint Sampling** (arXiv:2504.11713), Fig. 1: three distribution snapshots
  left→right (uncontrolled base → reciprocal projection → target Boltzmann μ),
  with the sampling notation printed under each snapshot.
* **Sequential Boltzmann Generators** (arXiv:2502.18462), Fig. 1: *"SBG uses
  annealed Langevin dynamics to transport proposal flow samples towards the
  target distribution."* — again a transport strip.

Cost: a strip cannot show *where several different losses attach*. All four of
those papers have essentially one training objective.

### Tradition B — "lettered sub-panels with one long caption"

Used when the method has several coupled pieces, which is our case.

* **EnFlow** (arXiv:2512.22597), Fig. 1 has **six** lettered sub-panels
  (a)–(f) — energy landscape, prior-work gap, the coupling idea, the training
  objective, the sampling-time guidance, and the downstream use — and a single
  ~150-word caption that walks the reader through them. This is currently the
  most common shape for "flow matching + an energy model" papers.

### Our choice: a hybrid — strip as the spine, lettered panels hanging off it

`fig1_method.tex` keeps Tradition A's strip as band **(a)** (prior noise →
flow-matching band with three molecule glyphs → `x_1`), then hangs two lettered
loss panels off it and puts the physics oracle underneath as a full-width bus:

```
(a)  prior  --->  flow band  t=0 ... t*=0.95 ... t=1  --->  x_1
        |                                      |
        | log p (dashed, reverse time)         | v_theta at t*
        v                                      v
(b) GLOBAL: FFJORD log p vs E/kT       (c) LOCAL: score vs F/kT
        ^                                      ^
        |  E/kT                                |  F/kT
(d)  ================  universal neural potential (OMol25 / eSEN)  ============
(e)  L = L_FM + λ1 L_force + λ2 L_energy   ⟹   p_θ ∝ exp(-E/kT)      + legend
```

Design decisions worth keeping:

1. **The wiring is deliberately crossing-free.** The energy panel is on the
   *left* because the FFJORD reverse-time ODE terminates at `t=0`, which is the
   left end of the strip; the force panel is on the *right* because the score is
   read out at `t*=0.95`, which is the right end. The oracle is a full-width bus
   *below* both, so it feeds each panel with a short vertical arrow. Every
   alternative arrangement I tried (oracle on the right, or force panel on the
   left) required at least one wire to cross another.
2. **The two losses are visually different kinds of object.** `L_energy` gets a
   potential-energy-surface inset with K dots on it (a *global shape* statement);
   `L_force` gets a single-atom inset with two nearly-parallel vectors (a *local
   gradient* statement). A reader who only looks at the pictures still gets
   "one term is about the whole landscape, the other about one point".
3. **The `t=1` pole is stated in the figure, not hidden.** The score formula has
   a `(1-t)^{-1}` factor; the axis is annotated at `t*=0.95` and the panel says
   why. Reviewers of score-based papers look for exactly this.
4. **The 83-element / transition-metal claim is carried by the glyph**, not by
   text: the `t=1` molecule glyph contains a green metal atom bonded to the ring.
   Cheap, and it pre-empts "does this only do CHNO?".
5. **`Z` cancels is printed inside the `L_energy` badge.** The variance form is
   the single least obvious step of the method; a reader who misses it thinks we
   need a partition function.
6. **No `\includegraphics`, no PNG of a rendered molecule.** Pure TikZ means the
   figure is diffable, recompiles anywhere, and has no raster artefacts at print
   resolution. It also means the in-figure font *is* the paper font.

Implementation constraints discovered on this machine (TeX Live 2018,
pgf 3.0.1): `standalone.cls`, `pgfplots.sty`, `preview.sty` and the
`arrows.meta` TikZ library are **not installed**. `fig1_method.tex` therefore
uses only `tikz` + `arrows` (`>=stealth'`) + `positioning,calc,fit,backgrounds,`
`decorations.pathmorphing,shapes.geometric`, and self-bootstraps a standalone
`article` preamble guarded by `\ifx\tikzpicture\undefined`, so the same file
both compiles alone (`pdflatex fig1_method.tex`) and `\input`s into `main.tex`.
Verified both ways.

---

## 2. What experiment figures look like in this subfield

### 2a. The `log p` vs energy scatter (our headline metric)

Most Boltzmann-generator papers **do not** plot this. They plot *energy
histograms* of samples vs MD reference (TBG Fig. 1b/2d, SBG Figs. 3–6), free
energy profiles along a reaction coordinate, and ESS bar charts (TBG Fig. 4a).
Correlation between model likelihood and target energy is reported numerically
(NLL / ESS in tables, `value ± std` over 3–5 runs) rather than drawn.

The one directly relevant precedent is **EnFlow Fig. 4**, whose caption is worth
copying the structure of verbatim:

> *"Molecule-level alignment between learned energy scores and single-point
> GFN2-xTB energies. Scatter plots show learned energy scores versus
> single-point GFN2-xTB energies for representative GEOM-Drugs molecules. Each
> point corresponds to a generated conformation, and colors indicate
> learned-energy rank from low to high. **Energies are shifted within each
> molecule so that the lowest-energy generated conformation has zero relative
> energy.** For visualization clarity, a small number of extreme high-energy
> outliers are omitted from the displayed panels; all correlations are computed
> using the full set of generated conformations. These examples show that the
> learned energy scores preserve rank-level conformational ordering across
> molecules with different correlation strengths and noise levels."*

Four transferable lessons, all adopted in `fig2_boltzmann_scatter`:

1. **Small multiples, one panel per parent molecule** — never one pooled cloud.
2. **Centre/shift within each molecule.** EnFlow shifts to the per-molecule
   minimum; we centre both axes on the per-parent mean. Same purpose, and for us
   it is not cosmetic: the per-molecule `log Z` spread is ~1e9, so a pooled plot
   is dominated by between-molecule offsets. This is the Simpson's paradox
   already flagged in `CLAUDE.md`, and it is why the metric is per-group.
3. **Pick parents spanning the real spread**, and say so ("different correlation
   strengths and noise levels"). Our script picks the 20th / 55th / 92nd
   percentile parent by measured `r`, so it can never be accused of showing only
   the best one.
4. **Colour by rank**, with a perceptually monotone sequential map. We use
   `cividis`, which is CVD-optimised (unlike `viridis`, whose green–yellow end
   loses ordering for some deuteranopes, and unlike `jet`, which is
   non-monotone in lightness).

Our addition, which no baseline paper has: **a fourth panel showing the
distribution of per-parent `r` over all 120 held-out parents for the energy
model vs the no-physics model, annotated with the pooled `r^2`.** This turns the
methodological caveat into evidence: pooled `r^2` is 0.11 (energy) and 0.01
(no physics) while the *mean per-parent* `r` is +0.43 vs +0.07. It simultaneously
(i) shows the full distribution rather than a mean, (ii) justifies the grouped
metric, and (iii) is entirely real data from
`runs/eval_ours/wide_*/boltz_independent.csv`.

### 2b. Ablation bars with SEM

Convention in the neighbourhood: TBG Fig. 4a is a grouped bar chart of ESS per
test peptide with one colour per architecture; FlowMol3 Fig. 4 is a row of bar
plots with a horizontal reference line for the training-data value and colour
families for diffusion / flow-matching / FlowMol baselines. Uncertainties in
these papers usually live in tables (`±std` over 3–5 seeds), not on the bars.

We deviate on one point on purpose: **with 3–4 seeds, plot the individual seed
points on top of the bar.** A bar + SEM whisker from n=4 invites the reader to
imagine a tight Gaussian; the raw dots show that the energy arm's spread is
driven by one high seed (0.525) and the scrambled arm's by one high seed
(0.417). Modern ML and biology reviewers now expect this ("show the n"), and in
our case it is *also* the honest way to present a result whose second-order
effect is not yet significant.

Other choices:
* **Hatching on top of colour** (`///` for the single-seed n=30 cells, `xxx` for
  the Y-scrambled negative control) so the chart survives greyscale printing and
  the colour is never the only channel.
* The two single-seed n=30 cells (`force only`, `force + energy`) are labelled
  `1 seed / n=30` under the axis. They come from a different, smaller evaluation
  round than the n=120 bars and must not be silently compared.
* `force only` (+0.109) sits *below* the no-physics baseline of its own round
  (+0.223 at n=30) — the negative result. The bar chart shows the number; the
  same-round baseline belongs in the text or a footnote, because putting two
  different `n` baselines in one chart is what creates false comparisons.

### 2c. Visualising Y-scrambling

The convention comes from QSAR, not from generative modelling: Rücker,
Rücker & Meringer, *"y-Randomization and Its Variants in QSPR/QSAR"*,
J. Chem. Inf. Model. 47(6) 2007 (and Rücker et al., *Y-Randomization — A Useful
Tool in QSAR Validation, or Folklore?*). The standard plot there is the
**distribution of `r^2` from many scrambled refits, with the real model's `r^2`
marked as a single reference line** — a permutation-test null.

We cannot honestly draw that plot: we have **3** scrambled seeds, not 100. Doing
a kernel density over 3 points would manufacture a null distribution that does
not exist. So `fig3` panel (b) instead:

* shows the two arms as paired bars **with all seed dots visible**,
* prints the measured contrast in black — `Δ = +0.092, t = 1.77, p = 0.147
  (n.s.)` at n = 120,
* prints the projected n = 240 contrast in **red, boxed, labelled
  PLACEHOLDER** — `Δ ≈ +0.13, t ≈ 3.9, p < 0.05`,
* and states in a footnote the fact that makes the control tight: the energy
  multiset and the within-parent energy std (77.6 kcal) are *identical* in both
  arms; only the geometry↔energy pairing is permuted.

If/when the n = 240 run gives ≥ 8 scrambled seeds, the right upgrade is to
switch panel (b) to the QSAR form (histogram of scrambled `r` + a vertical line
for the true-pairing mean) and cite Rücker et al. for the construction.

### 2d. Mechanism decomposition

No precedent in this subfield; the waterfall/bridge chart is the standard form
for "total effect = component + component" and reads instantly. `fig4` uses one
absolute bar (baseline), two floating bars (regularisation, Boltzmann
information), one absolute bar (total), dashed connectors, and a legend that
states the *operational* definition of each component ("survives Y-scrambling",
"destroyed by Y-scrambling") rather than a theoretical one.

---

## 3. Format / typography / colour rules

**ICLR is single-column.** The 2026 instructions (`iclr2026_conference.sty`)
specify a text block **5.5 in wide × 9 in tall, 10 pt type on 11 pt leading,
Times New Roman**. There is no 3.25 in two-column measure — do not size figures
for one. Practical widths:

| purpose | width |
|---|---|
| full text width | **5.50 in** (`FULL_W`) |
| two side by side | 2.65 in (`HALF_W`) |
| 2/3 width, single panel | 3.60 in (`TWO3_W`) |

* **Draw at final size; never rescale in LaTeX.** `\includegraphics[scale=0.7]`
  or a `\resizebox` around a `tikzpicture` shrinks the in-figure text below the
  caption size and is the single most common figure defect in submissions. All
  three matplotlib figures are emitted at exactly `FULL_W`/`TWO3_W`;
  `fig1_method.tex` is 13.95 cm = 5.49 in at `scale=1.0`.
* **Font sizes inside figures**: body 8 pt, tick labels 6.5–7 pt, annotations
  5.4–6.5 pt, against a 10 pt body and ~9 pt caption. Nothing below ~5.4 pt.
* **`pdf.fonttype = 42`** in the matplotlib rc so no Type-3 fonts reach arXiv
  (arXiv rejects/complains about Type-3, and ICLR PDFs get font-checked).
* **Palette: Okabe & Ito (2008)**, the standard CVD-safe qualitative set, used
  identically in the TikZ figure and the matplotlib figures so the paper reads
  as one system:

  | hex | name | meaning in this paper |
  |---|---|---|
  | `#0072B2` | blue | model / flow matching / learned quantities |
  | `#009E73` | bluish green | `L_energy`, global shape, the positive result |
  | `#D55E00` | vermillion | `L_force`, local gradient, the negative result |
  | `#E69F00` | orange | physics oracle (OMol25 / eSEN, DFT labels) |
  | `#CC79A7` | reddish purple | negative control (Y-scrambled) |
  | `#7F7F7F` | grey | no-physics baseline / neutral structure |
  | `#F0E442` | yellow | highlight band only (Y-scrambling region) |
  | `#E31A1C` | red | **PLACEHOLDER ONLY — never a data colour** |

  Green vs vermillion is the one pair a deuteranope may find harder; that is why
  `L_force` bars are hatched `///` and `L_energy` bars are solid, and why the
  two never appear as adjacent unhatched bars.
* Sequential ramp: **`cividis`** (CVD-optimised, monotone lightness).
* `axes.spines.top/right = False`, no gridlines, thin (0.6 pt) spines — matches
  the restrained look of FlowMol3/EnFlow result panels.

---

## 4. Files produced

| file | what |
|---|---|
| `paper/figures/fig1_method.tex` | Figure 1, pure TikZ. Compiles standalone (`pdflatex fig1_method.tex`, verified) **and** `\input`s into a 5.5 in host document (verified). |
| `paper/figures/make_experiment_figures.py` | fig2/fig3/fig4. Runs with `envs/omol25/bin/python` (matplotlib 3.10.8, numpy 2.4.4, scipy 1.17.1; **no pandas** in that env, so it uses `csv`). |
| `paper/figures/dump_boltz_records.py` | writes `<run>/boltz_records.csv` (per-perturbation `log p` + xTB energy) so fig2's scatter panels become fully real. Needs `xtb` on PATH. |
| `paper/figures/out/fig{2,3,4}_*.{pdf,png}` | generated output, regenerable. |

### How to include them in `main.tex`

`skeleton_min.tex` (the compile-verified ICLR 2026 skeleton in `paper/`) loads
`graphicx` and `xcolor` but **not** `tikz`, and does **not** define `\pending`.
Add both:

```latex
\usepackage{tikz}                                        % required by fig1
\providecommand{\pending}[1]{\textcolor{red}{#1}}        % red placeholder marker
```

then

```latex
\begin{figure}[t]\centering
  \input{figures/fig1_method}                 % pure TikZ, natural size 5.49in
  \caption{...}\label{fig:method}
\end{figure}

\begin{figure}[t]\centering
  \includegraphics[width=\linewidth]{figures/out/fig3_ablation_bars.pdf}
  \caption{...}\label{fig:ablation}
\end{figure}
```

Verified: injecting fig1 (`\input`) and fig2/fig3 (`\includegraphics`) into the
real `skeleton_min.tex` compiles with **zero errors and zero Overfull hbox**,
i.e. all four figures fit the 5.5 in measure exactly. `\pending{}` renders red
inside a caption.

Two small caveats worth knowing:

* matplotlib is configured with `savefig.bbox="tight"`, so the emitted PDFs come
  out 0.5–2% narrower than the nominal `FULL_W`. `width=\linewidth` therefore
  scales them *up* by <2%, which is imperceptible; do not use any other width.
  Never wrap `\input{figures/fig1_method}` in a `\resizebox`.
* In `fig1_method.tex` the smallest text uses `\bgfmtiny`
  (`\fontsize{6}{6.6}`), deliberately one step above `\tiny` (= 5 pt at 10 pt
  base), which is below the conventional 6 pt floor for in-figure type. If a
  layout ever needs to shrink, change that one macro rather than individual
  nodes — but re-check the two `text width` notes in panels (b) and (c), which
  are tuned to exactly two/three lines at 6 pt.

Regenerate everything:

```bash
cd /n/home04/yulili/bgfm/paper/figures
pdflatex fig1_method.tex
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python make_experiment_figures.py
# once the per-record file exists, fig2 loses its red stamp:
/n/holylabs/woo_lab/Lab/yulili/bgfm/envs/omol25/bin/python make_experiment_figures.py --only fig2
# camera-ready gate: non-zero exit if any placeholder is still in a figure
... make_experiment_figures.py --strict-real
```

All numbers the script prints were cross-checked against
`runs/eval_ours/*/boltz_independent.json` on 2026-08-03 and agree to 4 decimals
(`wide_a1_fm_only_s{2,3,4,5}` → 0.0750/0.0829/0.0647/0.0771;
`wide_a3_energy_only_s{1,2,3,5}` → 0.3768/0.4025/0.5246/0.4181;
`wide_a6_*shuffled*_s{2,3}`, `*_stab_s5` → 0.2843/0.3155/0.4172). The script
warns loudly on stderr if disk ever disagrees with the hard-coded values.

---

## 5. Caption drafts

Captions are written to be self-contained (a reviewer who reads only figures and
captions should get the paper), and they obey the wording discipline: no
"zero-shot to unseen elements", no "beats all baselines", negative results
stated, placeholders flagged.

### Figure 1 — method overview

> **Figure 1: Boltzmann-Guided Flow Matching.** **(a)** Generation is
> SE(3)-equivariant flow matching: an isotropic Gaussian point cloud `x_0`
> together with a CTMC prior over atom types and charges is transported by
> `dx_t/dt = v_θ(x_t,t)` to a 3D structure `x_1` given as positions and atomic
> numbers over 83 elements, with **no bond labels at any point** (bonds, when
> needed for evaluation, are recovered post hoc with xyz2mol). Reversing the
> same ODE (dashed) and accumulating the divergence with a Hutchinson estimator
> gives the exact log-density `log p_θ(x_1)` (FFJORD). **(b)** The
> *global-shape* term compares that log-density to the potential: for `K`
> geometric perturbations of a single parent molecule we penalise
> `Var_k[log p_θ(x_1^(k)) + E(x_1^(k))/kT]`. Because it is a variance, the
> per-molecule partition function cancels exactly, and because it is taken
> *within* a parent it is not contaminated by the ~1e9 spread of `log Z` across
> molecules. **(c)** The *local-gradient* term uses the closed-form score of the
> flow-matching marginal, `s_θ = (t v_θ − x_t)/((1−t)σ²)`, and matches it to the
> DFT force, `‖s_θ − F(x_1)/kT‖²`; at the Boltzmann optimum
> `∇log p = −∇E/kT = F/kT`. The score has a pole at `t = 1`, so it is read out
> at `t* = 0.95` with a per-atom norm cap. **(d)** Both terms are supervised by
> one universal neural potential — OMol25 / eSEN, 3.9M molecules, 205M atoms,
> 83 elements, DFT forces *and* energies, no bond labels. **(e)** The total
> objective, and the consistency statement it targets.

### Figure 2 — per-parent Boltzmann agreement

> **Figure 2: the model's exact log-density tracks an independent physics
> oracle, per molecule.** For each held-out molecule we generate `K` perturbed
> geometries (`σ = 0.15 Å`), compute `log p_θ` exactly by reverse-time
> integration, and compare against `−E/kT` from **GFN2-xTB**, a semi-empirical
> method that is *independent of the eSEN oracle used for training*. **Left
> three panels:** three representative parents at the 20th, 55th and 92nd
> percentile of the measured per-parent Pearson `r`, so the panels span the real
> spread rather than showing only successes. Both axes are centred within each
> parent; points are coloured by Boltzmann rank; the dashed line is the
> per-parent least-squares fit. **Right:** the distribution of per-parent `r`
> over all 120 held-out parents, for the energy-trained model (green) and the
> no-physics flow-matching control (grey); dotted lines mark the means
> (+0.43 vs +0.07). Pooling all parents into one regression destroys the signal
> (`r² = 0.11` and `0.01`): between-molecule `log Z` offsets dominate, which is
> why the metric is defined within parents. *[Red note in the current build: the
> individual point positions in the three left panels are placeholders drawn to
> reproduce their real per-parent `r` and slope; run `dump_boltz_records.py` to
> make them the measured points.]*

### Figure 3 — ablation

> **Figure 3: the energy term works, the force term does not.** Mean per-parent
> Boltzmann `r` against GFN2-xTB. Bars are means over independently trained
> seeds, error bars are SEM, and every seed is plotted as an open circle
> (`n = 3–4` seeds; with so few seeds the raw points are more informative than
> the whisker). **(a)** Adding the energy-consistency term with true
> geometry–energy pairing raises `r` from +0.075 to +0.430
> (`Δ = +0.356`, Welch `t = 10.9`, `p ≈ 0.0015`) — the paper's main result. The
> force term is *harmful*: on its own it reaches only +0.109, and adding it to
> the energy term lowers the energy-only result (+0.294 vs +0.420 in the same
> evaluation round). The two force cells come from an earlier, smaller round
> (1 seed, `n = 30` parents) and must be compared only to that round's own
> baseline (+0.223), not to the `n = 120` bars. **(b)** Y-scrambling control:
> permuting the geometry↔energy pairing *inside* each parent leaves the energy
> multiset and the within-parent energy std (77.6 kcal) untouched, so any drop
> isolates genuine Boltzmann information. Measured at `n = 120` the gap is
> `Δ = +0.092`, `t = 1.77`, `p = 0.147` — **not yet significant**; a larger
> `n = 240` evaluation is in progress and the projected outcome is shown in red
> as a placeholder.

### Figure 4 — mechanism decomposition

> **Figure 4: what the energy term actually buys.** Decomposition of the
> +0.356 total improvement using the Y-scrambling control as the pivot: the part
> that survives label scrambling is generic density regularisation (the loss
> penalises high-variance densities regardless of which energy goes with which
> geometry), and the part that is destroyed by scrambling is Boltzmann
> information (the model actually using `E(x)` as a function of `x`). The split
> is ≈63% / ≈37%. *[Red note in the current build: this split is computed from
> the 2-seed Y-scrambled mean (0.300); with all three scrambled seeds on record
> (0.339) it becomes 74% / 26%. It must be recomputed, and the main text made to
> agree, once the `n = 240` scrambled seeds land — see `--split seeds`.]*

---

## 6. Placeholder inventory for these figures

Mirror this into the manuscript's `PLACEHOLDER INVENTORY` block. Every entry is
rendered in red and boxed in the figure, and `make_experiment_figures.py` prints
the same list at the end of every run (and exits non-zero under
`--strict-real`).

| # | figure | placeholder | true current status |
|---|---|---|---|
| P-F1 | fig2, left panels | the individual `(log p, −E/kT)` point positions | the per-parent `r` and OLS slope of each panel are **real**; only the points are synthesised to reproduce them. Fixed by `dump_boltz_records.py`. |
| P-F2 | fig3(b) | projected `n = 240` Y-scrambling: `Δ ≈ +0.13`, `t ≈ 3.9`, `p < 0.05` | measured at `n = 120`: `Δ = +0.092`, `t = 1.77`, `p = 0.147`, **not significant**. n=240 eval running. |
| P-F3 | fig4 | the 63% / 37% regularisation / Boltzmann split | derived from the **2-seed** scrambled mean (0.2999). With all 3 scrambled seeds (0.3390) the split is 74% / 26%. **The 63/37 figure quoted in the project fact sheet and the 3-seed data on disk are inconsistent — this must be resolved before submission.** |

## 7. Figures still to build (not in this deliverable)

* **OOD / transition-metal figure (tmQM).** Real data exists
  (`runs/eval_ours/OOD_tmqm_*`): force-only `r = −0.158` → energy `r = +0.365`
  (median +0.429, 37% of complexes above 0.5), 270/270 xTB single points
  converged so there is no survivor bias. Suggested form: per-metal (Cr, Fe, Mn,
  Mo, Ru, Ti, V) strip/violin of per-parent `r` for force-only vs energy, with
  the training-set atom share printed per metal (each metal is 1.5k–51k atoms,
  0.34% of the 205M training atoms). **Caption must say "sparse tail plus
  distribution shift / transfer to a different transition-metal dataset", never
  "zero-shot to unseen elements"** — OMol25 does contain these metals.
* **Numerical-stability figure.** `energy_b_parents = 1` gives a single-parent
  variance estimate whose value jumped 48 → 974 between steps, producing
  gradient spikes and NaN weights in 3 of 9 seeds, and the old NaN guard zeroed
  the loss and *kept training*, hiding the failure for about a GPU-day.
  Suggested form: two-panel loss/weight-norm trace, diverging seeds in
  vermillion, plus the fix (`b_parents = 8`, `energy_loss_cap = 1500`,
  `FiniteWeightGuard`) with 2/2 seeds stable. This is a real engineering
  contribution and reviewers reward it.
* **GEOM COV/MAT context table (not a figure).** Per the project decision this
  is presented as *context* with an explicit task-mismatch statement (those 22
  baselines do conditional conformer generation from a molecular graph; we are
  an unconditional bond-free de-novo generator; measured AMR-R 3.03 Å at 50k vs
  0.073 Å) — do not draw it as a comparison chart, because a bar chart implies a
  fair comparison that does not exist.
