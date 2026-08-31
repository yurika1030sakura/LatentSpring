#!/bin/bash
# ===========================================================================
#  build.sh --- one-shot build of the BGFM ICLR 2026 submission.
#
#    ./build.sh            build main.tex
#    ./build.sh clean      remove LaTeX build artefacts (keeps the PDF)
#    ./build.sh distclean  remove artefacts and the PDF
#
#  Drives the four passes by hand because latexmk is NOT installed on this
#  host (TEMPLATE_NOTES.md Sec. 2). Two pdflatex passes after bibtex are
#  required: cleveref + hyperref need a second fixpoint pass.
# ===========================================================================
set -uo pipefail
cd "$(dirname "$0")"

DOC=main

case "${1:-build}" in
  clean)
    rm -f $DOC.aux $DOC.bbl $DOC.blg $DOC.log $DOC.out $DOC.toc $DOC.fls \
          $DOC.fdb_latexmk _tmp_*.* _test*.*
    echo "cleaned build artefacts (PDF kept)."; exit 0 ;;
  distclean)
    rm -f $DOC.aux $DOC.bbl $DOC.blg $DOC.log $DOC.out $DOC.toc $DOC.pdf \
          _tmp_*.* _test*.*
    echo "cleaned build artefacts and the PDF."; exit 0 ;;
esac

# --- toolchain check -------------------------------------------------------
missing=0
for tool in pdflatex bibtex; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: '$tool' not found on PATH." >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF'

No LaTeX toolchain is available in this shell. Options, in order of preference:

  1. On a FASRC node the system TeX Live is usually already on PATH:
         which pdflatex bibtex
     If it is not, try:
         module load texlive          # or: module avail texlive
  2. Build the PDF elsewhere. The directory is self-contained apart from four
     package that lives in TEXMFHOME (=$HOME/texmf) on this host and is NOT
     part of the submission tarball:
         cleveref
     (multirow and algorithmicx were dropped on 2026-08-12 -- they were loaded
     and never used; threeparttable was never loaded at all.)
     Overleaf and any full TeX Live ship cleveref, so uploading this
     directory as-is is enough.
  3. Everything else the build needs (iclr2026_conference.{sty,bst},
     math_commands.tex, natbib.sty, fancyhdr.sty) is vendored in this
     directory, so no network access is required.

EOF
  exit 127
fi

# --- required inputs -------------------------------------------------------
for f in sections/01_intro.tex sections/02_method.tex sections/03_theory.tex \
         sections/04_experiments.tex sections/05_conclusion.tex \
         sections/A1_proofs.tex sections/A2_details.tex refs.bib \
         figures/fig1_objective_float.tex figures/tabA2_constants.tex; do
  [ -f "$f" ] || echo "WARNING: missing input '$f' (build will continue)." >&2
done
# fig4_mechanism_decomposition was DELETED 2026-08-04 (SPINE_v2 ruling R4d);
# do not re-add it to this list.
for f in figures/out/fig2_ordering.pdf figures/out/fig3_grid.pdf \
         figures/out/fig4_calibration.pdf; do
  [ -f "$f" ] || echo "WARNING: missing figure '$f'; regenerate with its
    generator script in figures/." >&2
done

# --- four passes -----------------------------------------------------------
echo "==> pdflatex (1/3)"; pdflatex -interaction=nonstopmode -halt-on-error $DOC >/dev/null || {
  echo "pdflatex pass 1 FAILED -- last errors:" >&2
  grep -A3 -m20 '^!' $DOC.log >&2; exit 1; }
echo "==> bibtex";         bibtex $DOC >/dev/null 2>&1 || echo "  (bibtex reported problems; see $DOC.blg)"
echo "==> pdflatex (2/3)"; pdflatex -interaction=nonstopmode $DOC >/dev/null
echo "==> pdflatex (3/3)"; pdflatex -interaction=nonstopmode $DOC >/dev/null

# --- report ----------------------------------------------------------------
echo
echo "---------------------------------------------------------------"
MAINPAGES=$(sed -n 's/.*newlabel{endofmaintext}{{[^}]*}{\([0-9]*\)}.*/\1/p' $DOC.aux)
LIMIT=9
if [ -n "$MAINPAGES" ] && [ "$MAINPAGES" -le "$LIMIT" ]; then
  echo "MAIN TEXT PAGES: $MAINPAGES  (ICLR hard limit: $LIMIT)  -- PASS"
else
  echo "MAIN TEXT PAGES: $MAINPAGES  (ICLR hard limit: $LIMIT)  -- *** FAIL: OVER THE LIMIT ***"
fi
echo "TOTAL PAGES:     $(pdfinfo $DOC.pdf 2>/dev/null | sed -n 's/^Pages: *//p')"
echo "PLACEHOLDERS:    $(grep -h '\\pending{' main.tex sections/*.tex 2>/dev/null | grep -v '^[[:space:]]*%' | grep -o '\\pending{' | wc -l) live occurrences of \\pending{} (gate: 0)"
echo "                 (comment lines excluded; figures/ is not scanned -- it is owned"
echo "                  by the figure generators.)"
echo "FORBIDDEN VOCAB: $(pdftotext -layout $DOC.pdf - 2>/dev/null | grep -oiE 'in progress|placeholder|TODO|TBD|retraction|we retract|withdraw|earlier version|surprisingly|calibrated density|zero-shot|global Boltzmann|outperform all' | wc -l) hits in the RENDERED pdf (gate: 0)"
echo "---------------------------------------------------------------"
echo "LaTeX errors:"
grep -c '^!' $DOC.log 2>/dev/null || true
grep -m10 '^!' $DOC.log 2>/dev/null
echo "Undefined references / citations:"
grep -c 'undefined' $DOC.log 2>/dev/null || true
grep -o 'Reference `[^'"'"']*' $DOC.log 2>/dev/null | sort -u | head -20
grep -o "Citation \`[^']*" $DOC.log 2>/dev/null | sort -u | head -20
echo "Overfull boxes (>5pt):"
grep -o 'Overfull \\hbox ([0-9.]*pt' $DOC.log 2>/dev/null | grep -o '[0-9.]*pt' \
  | awk -F pt '$1>5' | wc -l
echo "Overfull boxes (any size):"
grep -c 'Overfull' $DOC.log 2>/dev/null || true
echo "---------------------------------------------------------------"
echo "Output: $(pwd)/$DOC.pdf"
