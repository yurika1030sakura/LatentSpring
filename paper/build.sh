#!/bin/bash
# Compile in /tmp by default; fail on compilation, citation or page-limit errors.
# Usage: bash paper/build.sh [build-directory]
set -euo pipefail
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${1:-${BGFM_PAPER_BUILD_DIR:-/tmp/bgfm-paper-build}}"
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
for tool in pdflatex bibtex python; do
  command -v "$tool" >/dev/null || { echo "Missing tool: $tool" >&2; exit 127; }
done
cd "$SOURCE_DIR"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" main.tex > "$BUILD_DIR/pass1.log"
(
  cd "$BUILD_DIR"
  BIBINPUTS="$SOURCE_DIR:" BSTINPUTS="$SOURCE_DIR:" bibtex main > bibtex.log
)
for pass in 2 3; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$BUILD_DIR" main.tex > "$BUILD_DIR/pass${pass}.log"
done
python - "$BUILD_DIR" <<'CHECK'
import re,sys
from pathlib import Path
root=Path(sys.argv[1])
aux=(root/'main.aux').read_text()
log=(root/'main.log').read_text()
m=re.search(r'\\newlabel\{endofmaintext\}\{\{[^}]*\}\{(\d+)\}',aux)
errors=[]
if not m:
    errors.append('Missing endofmaintext page marker')
else:
    pages=int(m.group(1)); print(f'Main text: {pages}/9 pages')
    if pages>9: errors.append('Main text exceeds the ICLR submission limit')
if re.search(r'(Citation|Reference).*undefined|There were undefined references',log):
    errors.append('Undefined citations or references')
if re.search(r'^!',log,re.M): errors.append('LaTeX error in final log')
print(f'PDF: {root / "main.pdf"}')
if errors:
    raise SystemExit('\n'.join(errors))
print('Build checks passed. This verifies formatting, not scientific readiness.')
CHECK
