"""PDF text extraction and layout cleanup.

The MaRisk circulars are typeset as two columns: the binding requirement on the left,
numbered by Textziffer, and the Erläuterungen on the right, aligned with the Textziffer
they explain. pdfplumber's default reading order runs across both columns line by line,
which splices half a requirement onto half an explanation. This module therefore

1. finds the column gap per page (drawn vertical rule if there is one, else the widest
   vertical band without words),
2. reads each column separately and re-attaches every explanation block to the
   requirement block it is vertically aligned with,
3. strips running headers/footers and bare page numbers,
4. excludes table-of-contents pages, whose entries match the section-heading pattern.

Every step is reported by ingest_pdf() so that its effect can be inspected rather than
trusted.
"""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from regrag import config
from regrag.text import clean_text

_MARKERS = "|".join(re.escape(m) for m in sorted(config.STRUCTURE_MARKERS, key=len, reverse=True))
# Same pattern as scripts/probe_extraction.py: marker, whitespace, digit.
HEADING_RE = re.compile(rf"^({_MARKERS})\s+\d")
_TOC_ENTRY_RE = re.compile(rf"^({_MARKERS})\s+\d[\d.]*\s+.*?(\.{{3,}}|\s)\s*\d{{1,3}}$")
_TEXTZIFFER_START_RE = re.compile(r"^\d{1,2}\s+\S")
_PAGE_NUMBER_RE = re.compile(r"^(seite\s+)?\d{1,3}(\s*(von|/)\s*\d{1,3})?$", re.IGNORECASE)


@dataclass
class Page:
    doc: str      # file name, e.g. "marisk_2024.pdf"
    page: int     # 1-based page number in the PDF
    text: str
    layout: str = "plain"   # how the page was read: "plain" or "columns"


@dataclass
class IngestReport:
    doc: str
    n_pages: int = 0
    empty_pages: list[int] = field(default_factory=list)
    toc_pages: list[int] = field(default_factory=list)
    column_pages: list[int] = field(default_factory=list)
    furniture_lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# --- Column detection -----------------------------------------------------

def find_column_split(words: list[dict], page_width: float, rules: list[float] = ()) -> float | None:
    """x coordinate separating the two columns, or None for a single-column page.

    `words` are pdfplumber word dicts (x0, x1, top, text). `rules` are x
    positions of long vertical lines drawn on the page; a rule inside the search band
    is the most reliable signal and wins if both sides of it carry text.
    """
    if len(words) < 20:
        return None
    lo, hi = (f * page_width for f in config.COLUMN_SEARCH_BAND)
    min_side = config.COLUMN_MIN_SIDE_SHARE * len(words)

    def balanced(x: float) -> bool:
        left = sum(w["x1"] <= x for w in words)
        right = [w for w in words if w["x0"] >= x]
        # At least three lines on the right: a right-aligned date or page number is not a column.
        right_lines = len({round(w["top"]) for w in right})
        return left >= min_side and len(right) >= min_side and right_lines >= 3

    for x in sorted(rules, key=lambda r: abs(r - page_width / 2)):
        if lo <= x <= hi and balanced(x):
            return float(x)

    # Whitespace gap: count how many words cover each 1-pt slot of the band. Headings
    # run across both columns, so a slot is "empty" if only a handful of words cover it.
    width = int(page_width) + 1
    cover = [0] * width
    for w in words:
        for x in range(max(0, int(w["x0"])), min(width, int(w["x1"]) + 1)):
            cover[x] += 1
    tolerance = max(2, int(0.02 * len(words)))
    best, run_start = None, None
    for x in range(int(lo), int(hi) + 2):
        empty = x <= int(hi) and cover[x] <= tolerance
        if empty and run_start is None:
            run_start = x
        elif not empty and run_start is not None:
            run_width = x - run_start
            if run_width >= config.COLUMN_GAP_MIN_WIDTH and (best is None or run_width > best[1]):
                best = (run_start, run_width)
            run_start = None
    if best is None:
        return None
    split = best[0] + best[1] / 2
    return split if balanced(split) else None


def _group_lines(words: list[dict], tolerance: float = 3.0) -> list[tuple[float, str]]:
    """Cluster words into lines by their top coordinate; returns (top, text) pairs."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tolerance:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [(ln[0]["top"], " ".join(w["text"] for w in sorted(ln, key=lambda w: w["x0"]))) for ln in lines]


def columns_to_text(words: list[dict], split: float) -> str:
    """Read a two-column page as requirement blocks, each followed by its explanation.

    The left column is cut into blocks at every line that starts a Textziffer or a
    section. Each right-column line is attached to the block whose vertical range it
    falls into. Right-column text above the first block continues an explanation from
    the previous page and is emitted first, where it joins the previous page's text.
    """
    left = [w for w in words if w["x0"] < split]          # straddling words (headings) go left
    right = [w for w in words if w["x0"] >= split]
    left_lines, right_lines = _group_lines(left), _group_lines(right)
    # The column title repeated on every page carries no content.
    right_lines = [(top, text) for top, text in right_lines if text.strip().lower() != "erläuterungen"]
    if not left_lines:
        return "\n".join(text for _, text in right_lines)

    anchors = [top for top, text in left_lines if HEADING_RE.match(text) or _TEXTZIFFER_START_RE.match(text)]
    if not anchors or anchors[0] > left_lines[0][0]:
        anchors.insert(0, left_lines[0][0])

    def block_of(top: float) -> int:
        # Index of the last anchor at or above this line; -1 = above the first block.
        idx = -1
        for i, a in enumerate(anchors):
            if top + 3.0 >= a:
                idx = i
        return idx

    left_blocks: dict[int, list[str]] = {}
    right_blocks: dict[int, list[str]] = {}
    for top, text in left_lines:
        left_blocks.setdefault(block_of(top), []).append(text)
    for top, text in right_lines:
        right_blocks.setdefault(block_of(top), []).append(text)

    parts: list[str] = []
    if -1 in right_blocks:
        parts.append("\n".join(right_blocks[-1]))
    for i in range(len(anchors)):
        if i in left_blocks:
            parts.append("\n".join(left_blocks[i]))
        if i in right_blocks:
            parts.append(config.EXPLANATION_PREFIX + " " + "\n".join(right_blocks[i]))
    return "\n".join(parts)


def vertical_rules(page) -> list[float]:
    """x positions of vertical lines spanning at least 30 % of the page height."""
    rules = []
    for edge in getattr(page, "edges", []):
        if edge.get("orientation") == "v" and (edge["bottom"] - edge["top"]) >= 0.3 * page.height:
            rules.append(edge["x0"])
    return rules


def fits_split(words: list[dict], split: float) -> bool:
    """Whether a page is consistent with a column split found on other pages.

    A page with a single short Erläuterung has too few right-hand words for its own gap
    to be detected, yet read in plain mode that Erläuterung is spliced into the
    requirement text line by line. Such a page fits the document's split if text exists
    right of it and (almost) nothing runs across it.
    """
    straddling = sum(w["x0"] < split < w["x1"] for w in words)
    right = sum(w["x0"] >= split for w in words)
    return right > 0 and straddling <= max(2, int(0.02 * len(words)))


def extract_page(page, layout: str = config.EXTRACTION_LAYOUT,
                 document_split: float | None = None) -> tuple[str, str]:
    """Text of one pdfplumber page and the layout actually used.

    `document_split` is the typical split of the other pages of the same document; it is
    used for pages whose own gap is not detectable but whose words are consistent with it.
    """
    if layout == "plain":
        return page.extract_text() or "", "plain"
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    split = find_column_split(words, page.width, vertical_rules(page))
    if split is None and document_split is not None and fits_split(words, document_split):
        split = document_split
    if split is None:
        # Also in forced "columns" mode: a page without a gap has nothing to split.
        return page.extract_text() or "", "plain"
    return columns_to_text(words, split), "columns"


def document_split(pdf) -> float | None:
    """Median column split over the pages where one is detected, or None."""
    splits = sorted(s for s in (find_column_split(p.extract_words(), p.width, vertical_rules(p))
                                for p in pdf.pages) if s is not None)
    return splits[len(splits) // 2] if splits else None


# --- Cleanup --------------------------------------------------------------

def _furniture_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def find_furniture(pages: list[str]) -> set[str]:
    """Normalised lines that recur at the top or bottom of most pages."""
    if len(pages) < 3:
        return set()
    n = config.FURNITURE_EDGE_LINES
    counts: Counter = Counter()
    for text in pages:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        # Section headings are never furniture, even when a document repeats the current
        # heading at the top of every page: the chunker handles that repetition, and
        # removing a heading here would merge two sections.
        edge = {_furniture_key(ln) for ln in lines[:n] + lines[-n:] if not HEADING_RE.match(ln.strip())}
        counts.update(edge)
    min_pages = config.FURNITURE_MIN_PAGE_SHARE * len(pages)
    return {key for key, c in counts.items() if c >= min_pages}


def strip_furniture(pages: list[str], furniture: set[str]) -> list[str]:
    cleaned = []
    n = config.FURNITURE_EDGE_LINES
    for text in pages:
        lines = text.split("\n")
        nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(nonblank[:n] + nonblank[-n:])
        # A bare number is only taken for a page number on the very first or last line:
        # further in, it is more likely a Textziffer that ended up on a line of its own.
        outer = {nonblank[0], nonblank[-1]} if nonblank else set()
        keep = []
        for i, ln in enumerate(lines):
            if i in edge and _furniture_key(ln) in furniture:
                continue
            if i in outer and _PAGE_NUMBER_RE.match(ln.strip()):
                continue
            keep.append(ln)
        cleaned.append("\n".join(keep))
    return cleaned


def is_toc_page(text: str) -> bool:
    """A page dominated by "AT 4.1 Title ..... 12" entries is a table of contents."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    entries = sum(bool(_TOC_ENTRY_RE.match(ln)) for ln in lines)
    if re.search(r"inhaltsverzeichnis", text, re.IGNORECASE) and entries >= 2:
        return True
    return entries >= config.TOC_MIN_ENTRIES and entries >= 0.4 * len(lines)


# --- Entry points ---------------------------------------------------------

def ingest_pdf(path: Path, layout: str = config.EXTRACTION_LAYOUT) -> tuple[list[Page], IngestReport]:
    import pdfplumber  # imported here so that the pure-text functions above work without it

    report = IngestReport(doc=path.name)
    raw: list[tuple[str, str]] = []
    with pdfplumber.open(path) as pdf:
        split = document_split(pdf) if layout != "plain" else None
        for page in pdf.pages:
            raw.append(extract_page(page, layout, split))
    report.n_pages = len(raw)

    texts = [t for t, _ in raw]
    furniture = find_furniture(texts)
    report.furniture_lines = sorted(furniture)
    texts = strip_furniture(texts, furniture)

    pages = []
    for number, (text, (_, used_layout)) in enumerate(zip(texts, raw), start=1):
        text = clean_text(text)
        if not text:
            report.empty_pages.append(number)
            continue
        if is_toc_page(text):
            report.toc_pages.append(number)
            continue
        if used_layout == "columns":
            report.column_pages.append(number)
        pages.append(Page(doc=path.name, page=number, text=text, layout=used_layout))
    return pages, report


def ingest_corpus(paths: list[Path], layout: str = config.EXTRACTION_LAYOUT) -> tuple[list[Page], list[IngestReport]]:
    pages, reports = [], []
    for path in paths:
        p, r = ingest_pdf(path, layout)
        pages.extend(p)
        reports.append(r)
    return pages, reports


def corpus_paths(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    return sorted(raw_dir.glob("*.pdf"))
