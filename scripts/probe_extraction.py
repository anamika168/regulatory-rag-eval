#!/usr/bin/env python3
"""
Recon pass over BaFin MaRisk circulars, before committing to a chunking strategy.

Reports per document:
  1. Page count + characters extracted per page (0 chars => no text layer => OCR territory)
  2. Marker frequency, counted two ways: literal substring vs line-initial heading
  3. Raw dump of the first 800 chars of page 5, to inspect reading order and page furniture
  4. Per page, the column split found by regrag.ingest (x position, or "-" for one column),
     and page 5 again as the column-aware reader returns it, for side-by-side comparison

The two counts answer different questions. Literal counts are NOT mutually exclusive --
"BTO" and "BTR" both contain "BT" -- so they must never be summed. Heading counts are.

Known limitation: heading counts still include table-of-contents entries, which match the
same pattern. Front matter will be excluded at ingest.

Run:
    python scripts/probe_extraction.py
"""
import re
from collections import Counter

import pdfplumber

from regrag import config
from regrag.ingest import extract_page, find_column_split, vertical_rules

PDFS = sorted(config.RAW_DIR.glob("marisk_*.pdf"))
MARKERS = config.STRUCTURE_MARKERS

if not PDFS:
    raise SystemExit(f"No PDFs found in {config.RAW_DIR}")

# Longest marker first, so BTO/BTR match before the shorter BT prefix.
_alternatives = "|".join(re.escape(m) for m in sorted(MARKERS, key=len, reverse=True))
# A heading is: start of line, marker, whitespace, digit -- e.g. "BTO 2.1 Funktionstrennung"
HEADING_RE = re.compile(rf"^({_alternatives})\s+\d", re.MULTILINE)

for path in PDFS:
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
        splits = [find_column_split(p.extract_words(), p.width, vertical_rules(p)) for p in pdf.pages]
        found = sorted(x for x in splits if x is not None)
        doc_split = found[len(found) // 2] if found else None     # as regrag.ingest.document_split
        page5_columns = extract_page(pdf.pages[4], "columns", doc_split)[0] if len(pdf.pages) >= 5 else ""
    full = "\n".join(pages)

    print(f"\n===== {path.name} =====")
    print(f"pages: {len(pages)}")
    print(f"column split: {doc_split:.1f} (median over {len(found)} pages)"
          if doc_split is not None else "column split: none detected")

    headings = Counter(HEADING_RE.findall(full))
    print(f"\n{'marker':<10}{'literal':>10}{'headings':>10}")
    for m in MARKERS:
        print(f"{m:<10}{full.count(m):>10}{headings[m]:>10}")

    for i, (t, split) in enumerate(zip(pages, splits), 1):
        col = f"split x={split:6.1f}" if split is not None else "split      -"
        print(f"  page {i:>3}: {len(t):>6} chars   {col}" + ("   <-- NO TEXT LAYER" if not t.strip() else ""))

    print("\n--- page 5, first 800 chars ---")
    print(pages[4][:800] if len(pages) >= 5 else "(fewer than 5 pages)")
    print("\n--- page 5, column-aware reading, first 800 chars ---")
    print(page5_columns[:800] if len(pages) >= 5 else "(fewer than 5 pages)")