#!/bin/bash

# IEEE Conference Paper Compilation Script
# Handles LaTeX + BibTeX compilation with proper cross-reference resolution

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

MAIN_FILE="main"
PDF_OUTPUT="${MAIN_FILE}.pdf"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IEEE Paper Compilation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: First LaTeX pass (generates .aux files)
echo -e "${YELLOW}[1/5] First pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ First pass completed${NC}"
else
    echo -e "${RED}✗ First pass failed${NC}"
    pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex" | tail -20
    exit 1
fi

# Step 2: BibTeX (processes bibliography)
echo -e "${YELLOW}[2/5] Running BibTeX...${NC}"
if [ -f "references.bib" ]; then
    bibtex "${MAIN_FILE}" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ BibTeX completed${NC}"
    else
        echo -e "${RED}✗ BibTeX failed (this is OK if no citations)${NC}"
        bibtex "${MAIN_FILE}" 2>&1 | tail -10
    fi
else
    echo -e "${YELLOW}⊘ No references.bib found, skipping BibTeX${NC}"
fi

# Step 3: Second LaTeX pass (incorporates bibliography)
echo -e "${YELLOW}[3/5] Second pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Second pass completed${NC}"
else
    echo -e "${RED}✗ Second pass failed${NC}"
    pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex" | tail -20
    exit 1
fi

# Step 4: Third LaTeX pass (resolves all cross-references)
echo -e "${YELLOW}[4/5] Third pdflatex pass...${NC}"
pdflatex -interaction=nonstopmode -file-line-error "${MAIN_FILE}.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Third pass completed${NC}"
else
    echo -e "${RED}✗ Third pass failed${NC}"
    pdflatex -interaction=nonstopmode "${MAIN_FILE}.tex" | tail -20
    exit 1
fi

# Step 5: Check for warnings
echo -e "${YELLOW}[5/5] Checking for issues...${NC}"
WARNINGS=$(grep -i "warning" "${MAIN_FILE}.log" | grep -v "Package hyperref Warning" | wc -l)
ERRORS=$(grep -i "^!" "${MAIN_FILE}.log" | wc -l)

if [ "$ERRORS" -gt 0 ]; then
    echo -e "${RED}✗ Found $ERRORS error(s)${NC}"
    grep -A 2 "^!" "${MAIN_FILE}.log" | head -20
    exit 1
elif [ "$WARNINGS" -gt 2 ]; then
    echo -e "${YELLOW}⚠ Found $WARNINGS warning(s) (check ${MAIN_FILE}.log)${NC}"
    grep -i "warning" "${MAIN_FILE}.log" | grep -v "Package hyperref" | head -5
else
    echo -e "${GREEN}✓ No critical issues${NC}"
fi

# Final output
echo ""
echo -e "${BLUE}========================================${NC}"
if [ -f "${PDF_OUTPUT}" ]; then
    PDF_SIZE=$(ls -lh "${PDF_OUTPUT}" | awk '{print $5}')
    PDF_PAGES=$(pdfinfo "${PDF_OUTPUT}" 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo -e "${GREEN}✓ Compilation successful!${NC}"
    echo -e "${GREEN}  Output: ${PDF_OUTPUT}${NC}"
    echo -e "${GREEN}  Size: ${PDF_SIZE}${NC}"
    echo -e "${GREEN}  Pages: ${PDF_PAGES}${NC}"
    
    # Check page count
    if [ "$PDF_PAGES" -gt 15 ]; then
        echo -e "${RED}⚠ WARNING: ${PDF_PAGES} pages exceeds 15-page limit!${NC}"
    else
        echo -e "${GREEN}  ✓ Within 15-page limit${NC}"
    fi
else
    echo -e "${RED}✗ PDF not generated${NC}"
    exit 1
fi
echo -e "${BLUE}========================================${NC}"

# Clean up auxiliary files (optional)
read -p "Clean auxiliary files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleaning auxiliary files...${NC}"
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.synctex.gz
    echo -e "${GREEN}✓ Cleanup complete${NC}"
fi

echo ""
echo -e "${BLUE}Done! Open ${PDF_OUTPUT} to view the paper.${NC}"
