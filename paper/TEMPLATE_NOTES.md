# ICLR 2026 LaTeX template — landed, read, and compile-verified

Prepared 2026-08-03 on a FASRC login/compute node. Everything below was verified
by actually compiling, not inferred. Target venue: **ICLR 2026**.

---

## 1. File manifest

All files are the **genuine ICLR 2026** template, unpacked from
`https://raw.githubusercontent.com/ICLR/Master-Template/master/iclr2026.zip`
(the official `ICLR/Master-Template` repo also has an `iclr2026/` directory with
byte-identical files). **No 2024/2025 fallback was needed — no "replace me"
placeholder exists anywhere.**

| File | Size | Role |
|---|---|---|
| `iclr2026_conference.sty` | 9,025 | **the** style file. Never edit (see §4). |
| `iclr2026_conference.bst` | 26,973 | BibTeX style. Use with `\bibliographystyle{iclr2026_conference}`. |
| `iclr2026_conference.tex` | 16,899 | ICLR's official shell/instructions source. Authoritative on page limits. |
| `iclr2026_conference.pdf` | 200,508 | Rendered instructions — visual reference for what a correct page looks like. |
| `iclr2026_conference.bib` | 629 | 3 demo entries (`Hinton06`, `Bengio+chapter2007`, `goodfellow2016deep`). |
| `math_commands.tex` | 12,284 | Optional *dlbook* notation. Loads `amsmath,amsfonts,bm`. Has traps — §6.2. |
| `natbib.sty` | 45,154 | Vendored, so the citation style is reproducible. |
| `fancyhdr.sty` | 20,521 | Vendored, used for the running head. |
| `skeleton_min.tex` | — | **Our** minimal skeleton, compile-verified. Copy to `main.tex`. |
| `skeleton_min.pdf` | — | Proof it compiles here. |
| `.gitignore` | — | LaTeX build artifacts. |

A stray `iclr2026_conference.log` shipped inside the zip (ICLR's own build log);
deleted to avoid confusion with our compile logs.

## 2. Build

Verified toolchain on this host: **pdfTeX 3.14159265-2.6-1.40.19 (TeX Live 2018)**,
`/usr/bin/pdflatex`, `/usr/bin/bibtex`, `/usr/bin/xelatex`. **`latexmk` is NOT
installed** — drive the passes manually:

```bash
cd /n/home04/yulili/bgfm/paper
pdflatex -interaction=nonstopmode main    # writes .aux with \citation keys
bibtex   main                             # resolves citations -> main.bbl
pdflatex -interaction=nonstopmode main    # pulls in .bbl, numbers refs
pdflatex -interaction=nonstopmode main    # settles cross-refs / cleveref / hyperref
```

Four commands, in that order. Two `pdflatex` passes after `bibtex` are required
because `cleveref` + `hyperref` need a second fixpoint pass. Output is **US
Letter, 612 x 792 pt** — confirmed with `pdfinfo`; do not switch to A4.

Rendering a page to look at it (no `latexmk`, but `pdftoppm` and `convert` exist):

```bash
pdftoppm -r 110 -png -f 1 -l 1 main.pdf /tmp/pg   # -> /tmp/pg-1.png
```

## 3. Verified minimal `main.tex`

This is the preamble of `skeleton_min.tex` verbatim (body trimmed), which compiles **clean (0
errors, 0 undefined references/citations)** on this host. Copy it and write into
it.

```latex
\documentclass{article}                     % ICLR = plain article + the .sty

% --- REQUIRED. No options. Never edit the .sty.
\usepackage{iclr2026_conference,times}

% --- Math. amsmath BEFORE math_commands.tex.
\usepackage{amsmath,amssymb,amsthm}
\let\amseqref\eqref            % save amsmath \eqref -> "(3)"
\input{math_commands.tex}      % optional dlbook notation (\va \mA \E \R \KL ...)
\let\eqref\amseqref            % math_commands clobbers \eqref; undo it (see §6.2)

% --- TeX Live base; safe everywhere.
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{subcaption}
\usepackage{algorithm}
\usepackage{microtype}         % legal, buys ~2% length back
% --- Installed into ~/texmf on this host (see §6.5): multirow, algpseudocode, cleveref
\usepackage{multirow}
\usepackage{algpseudocode}

% --- hyperref late. Plain \usepackage{hyperref} draws ugly colored BOXES
%     around every link; colorlinks is what real ICLR papers use.
\usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=green!45!black,
            urlcolor=blue!70!black]{hyperref}
\usepackage{url}
\usepackage{cleveref}          % OPTIONAL, must be after hyperref

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\theoremstyle{definition}
\newtheorem{assumption}{Assumption}
\newtheorem{definition}{Definition}
\theoremstyle{remark}
\newtheorem{remark}{Remark}

\title{Boltzmann-Guided Flow Matching for \\ Bond-Free 3D Molecular Generation}
\author{Anonymous}            % ignored while \iclrfinalcopy is commented out
%\iclrfinalcopy               % UNCOMMENT ONLY for camera-ready

\begin{document}
\maketitle                    % REQUIRED: also installs the running head

\begin{abstract}
One paragraph only.
\end{abstract}

\section{Introduction}
Parenthetical \citep{Bengio+chapter2007}; in-sentence \citet{Hinton06}.

% ... body ...

\subsubsection*{Author Contributions}   % optional, unnumbered
\subsubsection*{Acknowledgments}        % funding goes HERE, not in a title footnote

% ---- main text ends here; nothing below counts against the page limit ----
\label{endofmaintext}

\bibliography{references}               % your .bib, no extension
\bibliographystyle{iclr2026_conference}

\appendix
\section{Proofs}
\end{document}
```

Bibliography: keep ICLR's demo `iclr2026_conference.bib` untouched as a
reference and create your own `references.bib`; `\bibliographystyle` must stay
`iclr2026_conference`.

## 4. Hard constraints read out of `iclr2026_conference.sty`

Geometry and type (`.sty` lines 36–54, 173–182) — these are *facts about the
compiled page*, useful for estimating length:

| Constraint | Value | Source |
|---|---|---|
| Paper size | 8.5 x 11 in (US Letter) | `\paperheight/\paperwidth` |
| Text block | **5.5 in wide x 9.0 in tall**, single column | `\textwidth 5.5in`, `\textheight 9.0in` |
| Left margin | 1.5 in (`\oddsidemargin .5in` + 1 in default) | `.sty` L40 |
| Body type | **10 pt on 11 pt leading** | `\def\normalsize{...{11pt}\xpt}` |
| Typeface | Times (`\usepackage{...,times}`) | required by shell |
| Paragraphs | **no indent**, separated by 1/2 line | `\parindent 0pt`, `\parskip .5pc` |
| Title | 17 pt small caps, **left**-aligned (not centered) | `\LARGE\sc` + `\def\LARGE{...\xviipt}` |
| `\section` | 12 pt small caps, flush left | `{\large\sc\raggedright}` |
| `\subsection`/`\subsubsection` | 10 pt small caps, flush left | `{\normalsize\sc\raggedright}` |
| `\paragraph` | 10 pt **bold**, run-in (text continues on same line) | `{\normalsize\bf}`, negative afterskip |
| Abstract | centered small-caps "Abstract", quoted/indented 1/2 in both sides, **one paragraph** | `\renewenvironment{abstract}` |
| Page numbers | centered footer, automatic via `fancyhdr` | verified in PDF |
| Footnotes | rule 12 pc wide, at bottom of the page they appear on | `\footnoterule` |

Behavior worth knowing:

- **Anonymity is enforced by the style file.** With `\iclrfinalcopy` commented
  out, `\@maketitle` *ignores* `\author` entirely and prints "Anonymous authors
  / Paper under double-blind review", and the running head reads **"Under review
  as a conference paper at ICLR 2026"**. Verified. ICLR's shell states
  non-anonymous submissions are *rejected without review*.
- **`\iclrfinalcopy`** (put it in the preamble, before `\begin{document}`):
  verified to switch the head to "Published as a conference paper at ICLR 2026",
  drop the line-number ruler, and start honoring `\author`. Real names go in
  only at camera-ready.
- **Line-number ruler**: the `.sty` auto-stamps gray line numbers in the left
  margin in submission mode (`\AddToShipoutPicture`). Verified present. The
  `.sty`'s own comment says **do not refer to those numbers in your text** —
  they vanish in the final version.
- **`\maketitle` is mandatory**: `\lhead{...}` is set *inside* `\@maketitle`, so
  skipping it silently loses the running head.
- **Table of contents is disabled** (`\def\addcontentsline#1#2#3{}`). Do not add
  a `\tableofcontents`, in the appendix either — it will render empty. (Side
  effect: hyperref PDF bookmarks are suppressed. Harmless.)
- **`\And` / `\AND`** separate author blocks at camera-ready (`\And` = let LaTeX
  break, `\AND` = force a break).

## 5. Page limit and the 12-page instruction

**This is the one place where the brief conflicts with the venue rule.**
`iclr2026_conference.tex` L131, verbatim:

> "There will be a strict upper limit of **9 pages** for the main text of the
> initial submission, with unlimited additional pages for citations. This limit
> will be expanded to **10 pages** for rebuttal/camera ready."

So: 9 pages main text, hard, for the initial submission. References are
uncapped. The appendix sits **after** `\bibliography` (that is exactly how
ICLR's own shell orders `\bibliography` then `\appendix`) and is off the budget,
but reviewers are not obliged to read it.

Writing to 12 pages first is fine as an internal draft — the user asked for it —
but plan the compression from the start rather than discovering a 3-page
overshoot at the end:

1. Mark every subsection that is **appendix-movable** as you write it (proof
   details, hyperparameter tables, extra ablation cells, dataset construction).
   Moving a block below `\bibliography` is then a cut-and-paste, not a rewrite.
2. Keep figures/tables in separate `\input{}` files so a float can be demoted to
   the appendix without touching prose.
3. Check the budget every compile:
   ```bash
   sed -n 's/.*newlabel{endofmaintext}{{[^}]*}{\([0-9]*\)}.*/main text pages: \1/p' main.aux
   ```
   (works because of the `\label{endofmaintext}` placed just before
   `\bibliography` in the skeleton; verified to print the right page).
4. Length levers that are *legal*: `microtype`, tighter float sizing,
   `\vspace{-...}` used sparingly, moving content to the appendix, cutting
   words. Levers that are **grounds for rejection**: changing `\textwidth`,
   `\textheight`, font size, or `\parskip`. ICLR's shell L88–89: "Tweaking the
   style files may be grounds for rejection", and L353: "Do not change any
   aspects of the formatting parameters in the style files."

## 6. Traps found while verifying (each one cost a compile)

**6.1 `natbib` option clash.** The `.sty` already does
`\RequirePackage{natbib}` plus
`\setcitestyle{authoryear,round,citesep={;},aysep={,},yysep={;}}`. Verified:
`\usepackage[numbers]{natbib}` (or any option) → `! LaTeX Error: Option clash
for package natbib.` A bare `\usepackage{natbib}` is harmless but pointless.
**Just don't load it.** Consequence: the citation style is author-year and you
cannot make it numeric. Verified rendering: `\citep{Bengio+chapter2007}` →
"(Bengio & LeCun, 2007)", `\citet{Hinton06}` → "Hinton et al. (2006)",
`\citep{a,b}` → "(Hinton et al., 2006; Goodfellow et al., 2016)".

**6.2 `math_commands.tex` silently redefines `\eqref`.** It does
`\def\eqref#1{equation~\ref{#1}}` (L35), overwriting amsmath's, so `\eqref{eq:x}`
would print "equation 1" instead of "(1)" — no error, just wrong output
everywhere. The skeleton's `\let\amseqref\eqref` … `\let\eqref\amseqref` sandwich
fixes it; verified to render "(1)". Same file also `\def`s `\figref \secref
\algref \chapref \partref \ceil \floor` in that lowercase-prose style.

**6.3 Reserved macros.** `math_commands.tex` defines ~600 names. `\newcommand`
on any of them → `! Command \foo already defined`. Relevant to a BGFM paper:
`\E` (= `\mathbb{E}`), `\R` (= `\mathbb{R}`), `\Ls` (= `\mathcal{L}`), `\Var`,
`\Cov`, `\KL`, `\emp`, `\1`, `\eps`, `\sign`, `\Tr`, `\argmax`, `\argmin`,
`\softmax`, `\sigmoid`, `\softplus`, `\parents`, `\train`/`\valid`/`\test`,
`\pdata`/`\pmodel`, `\normltwo`/`\normlp`, plus the whole single-letter families
`\va…\vz` (bold vector), `\mA…\mZ` (bold matrix), `\tA…\tZ` (tensor),
`\sA…\sZ` (set), `\gA…\gZ` (graph), `\ra…\rz`/`\rva…\rvz`/`\rmA…\rmZ` (random),
and `\eva…`/`\emA…`/`\etA…` (elements). Prefer reusing `\E \R \Ls \va \mI` over
inventing clashing names. If you'd rather have a clean namespace, comment out the
`\input{math_commands.tex}` line — it is explicitly optional
(`iclr2026_conference.tex` L233) — and keep `\usepackage{amsmath,amssymb}`.

**6.4 Floats can land above the title.** In the verified build, a `[t]`
`algorithm` and a `[t]` `table` declared in section 1/2 floated to the top of
page 1, *above* the paper title. Cosmetically bad. On page 1 prefer `[h]`,
`[!t]`, or declare the float after the text that cites it.

**6.5 Packages missing from this host's TeX Live 2018.** Confirmed absent from
`/usr/share/texlive`: `multirow`, `wrapfig`, `cleveref`, `algpseudocode`
(algorithmicx), `threeparttable`, `siunitx`, `makecell`, `adjustbox`,
`algorithm2e`, `thmtools`, `bbm`, `dsfont`, `physics`, `pgfplots`.
I fetched from CTAN and installed into `TEXMFHOME = /n/home04/yulili/texmf`
(190 KB total, `mktexlsr` run, all found by `kpsewhich`):
`multirow`, `bigdelim`, `bigstrut`, `wrapfig`, `cleveref`, `algorithmicx`
(`algpseudocode`, `algcompatible`, …), `threeparttable`.
These live *outside* `paper/`, so they are not part of the submission tarball —
which is correct, since OpenReview takes a PDF and Overleaf ships all of them.
Still **missing and to be avoided** in the manuscript: `siunitx` (`\SI{}{}` —
write units by hand), `makecell`, `adjustbox`, `algorithm2e`, `pgfplots`,
`bbm`/`dsfont` (use `\mathbb{1}` or `math_commands`' `\1`).

**6.6 Available and verified working:** `amsmath amssymb amsthm booktabs
graphicx xcolor subcaption caption microtype algorithm algorithmic enumitem
colortbl tabularx float soul pifont tikz mathtools sfmath` (+ the §5.5
installs). `amsthm` theorem/proof, `booktabs`, `multirow`, `algpseudocode`,
`cleveref` (`\Cref` → "Equation (1)", "Theorem 1", "Section 2") and `microtype`
were all exercised in `skeleton_min.pdf`.

**6.7 Figure/table caption sides are opposite.** ICLR's rule (shell L184, L206):
figure caption goes **below** the figure; table caption goes **above** the table.
Captions are lower case except the first word and proper nouns. Both are
numbered consecutively.

**6.8 Pre-existing repo fragment won't compile as-is.**
`/n/home04/yulili/bgfm/notes/appendix_hbc.tex` uses `\begin{lemma}` and
`\begin{proposition}` but only declares `theorem`, `corollary`, `assumption`,
`remark` via `\newtheorem`. If it is `\input` into the appendix, either delete
its four `\newtheorem` lines and rely on the preamble declarations in §3 (which
include `lemma` and `proposition`), or add the two missing ones — otherwise
`! LaTeX Error: Environment lemma undefined.` It also needs `amsthm` for
`\begin{proof}` (present in §3).

## 7. Verification record

Run on this host, 2026-08-03:

- `pdflatex` x1 → 0 errors; `bibtex` → 0 errors, `iclr2026_conference.bst`
  resolved all 3 demo keys; `pdflatex` x2 → **0 errors, no undefined
  references, no undefined citations**. `Output written on skeleton_min.pdf
  (2 pages)`.
- `pdfinfo`: `Page size: 612 x 792 pts (letter)`, `PDF version 1.5`.
- Rendered page 1 and inspected visually: running head "Under review as a
  conference paper at ICLR 2026", gray line-number ruler in the left margin,
  17 pt small-caps left-aligned title, "Anonymous authors / Paper under
  double-blind review", small-caps centered Abstract indented both sides,
  small-caps numbered section heads, centered page number, author-year
  citations, `\eqref` → "(1)", `\Cref` → "Equation (1)"/"Theorem 1", multirow
  table, algorithm float, amsthm Theorem + Proof with QED box.
- Separate build with `\iclrfinalcopy` uncommented: 0 errors, head becomes
  "Published as a conference paper at ICLR 2026", ruler gone, `\author`
  honored.

**LaTeX is present on this machine and the template was really compiled — this
is not a paper-only check.**
