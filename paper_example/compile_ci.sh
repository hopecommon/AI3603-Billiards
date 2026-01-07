#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_FILE="main"

pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" >/dev/null
if [ -f "references.bib" ]; then
  bibtex "${MAIN_FILE}" >/dev/null || true
fi
pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" >/dev/null
pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" >/dev/null

echo "OK: built ${MAIN_FILE}.pdf"
