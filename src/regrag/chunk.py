"""Chunking: structure-aware on MaRisk section markers, and a fixed-size baseline.

Structure-aware chunking works on two levels:

* Sections start at a line-initial STRUCTURE_MARKER followed by a section number
  ("BTO 2.1 Funktionstrennung"). These are the only hard boundaries.
* Inside a section, text is packed by whole Textziffern up to CHUNK_SIZE characters.
  A Textziffer is only split (into overlapping windows) if it alone exceeds CHUNK_SIZE.

Textziffer numbers are bare digits at the start of a line, which is also what a line
starting with "3 Monate" or a footnote looks like. A line-initial number therefore only
counts as a Textziffer if it is the next number in sequence within its section (1 after
a heading, then 2, 3, ...). This one rule removes nearly all false positives without a
list of exceptions.

The fixed-size baseline ignores all of this and cuts character windows. Each fixed chunk
still records the section it starts in, but only for error analysis -- relevance in the
evaluation is decided by the text, never by this label.
"""

import re
from dataclasses import asdict, dataclass, field

from regrag import config
from regrag.ingest import Page

_MARKERS = "|".join(re.escape(m) for m in sorted(config.STRUCTURE_MARKERS, key=len, reverse=True))

# "BTO 2.1 Funktionstrennung" -> marker BTO, number 2.1, title "Funktionstrennung".
# The title must start with an upper-case letter: "AT 4.3.2 Tz. 3 gilt entsprechend" at
# the start of a line is a citation, not a heading.
SECTION_HEADING_RE = re.compile(
    rf"^(?P<marker>{_MARKERS})\s+(?P<number>\d+(?:\.\d+)*)\.?(?:\s+(?P<title>.*))?$"
)
_CITATION_TITLE_RE = re.compile(r"^(Tz\.|Textziffer|Satz|Abs\.|Absatz|Nr\.|und\b|bis\b|sowie\b|gilt\b)")
_TEXTZIFFER_RE = re.compile(r"^(?P<tz>\d{1,2})\s+\S")


@dataclass
class Section:
    doc: str
    section_id: str             # "BTO 2.1"; "PREAMBLE" for text before the first heading
    title: str
    path: list[str]             # headings of the parents, outermost first
    page_start: int
    page_end: int
    # (Textziffer number or None, text, first page, last page).
    # None = text before the first Textziffer of the section.
    paragraphs: list[tuple[int | None, str, int, int]] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return f"{self.section_id} {self.title}".strip()


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section_id: str
    section_title: str
    heading_path: list[str]
    page_start: int
    page_end: int
    textziffern: list[int]
    text: str                   # body text only; this is what relevance is judged on

    @property
    def embed_text(self) -> str:
        """Text that gets embedded and indexed: headings in front of the body."""
        if not config.PREPEND_HEADING or not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


# --- Section parsing ------------------------------------------------------

def parse_heading(line: str) -> tuple[str, str] | None:
    """(section_id, title) if the line is a section heading, else None."""
    m = SECTION_HEADING_RE.match(line.strip())
    if not m:
        return None
    title = (m.group("title") or "").strip()
    if title and (not title[0].isupper() or _CITATION_TITLE_RE.match(title)):
        return None
    if len(line) > 160:          # a heading is a title, not a paragraph
        return None
    return f"{m.group('marker')} {m.group('number')}", title


def _parent_ids(section_id: str) -> list[str]:
    """"AT 4.4.2" -> ["AT 4", "AT 4.4"]."""
    marker, number = section_id.split(" ", 1)
    parts = number.split(".")
    return [f"{marker} {'.'.join(parts[:i])}" for i in range(1, len(parts))]


def parse_sections(pages: list[Page]) -> list[Section]:
    """Split page texts into sections and Textziffern, document by document."""
    sections: list[Section] = []
    for doc in dict.fromkeys(p.doc for p in pages):
        doc_pages = [p for p in pages if p.doc == doc]
        titles: dict[str, str] = {}
        current = Section(doc, "PREAMBLE", "", [], doc_pages[0].page, doc_pages[0].page)
        expected_tz = 1
        title_pending = False
        buffer: list[str] = []
        buffer_tz: int | None = None
        buffer_page = buffer_last_page = doc_pages[0].page

        def flush():
            nonlocal buffer
            text = "\n".join(buffer).strip()
            if text:
                current.paragraphs.append((buffer_tz, text, buffer_page, buffer_last_page))
            buffer = []

        for page in doc_pages:
            for line in page.text.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                heading = parse_heading(stripped)
                if heading:
                    section_id, title = heading
                    if section_id == current.section_id:
                        # Running header repeating the current heading on a new page.
                        continue
                    flush()
                    sections.append(current)
                    path = [f"{pid} {titles[pid]}".strip() for pid in _parent_ids(section_id) if pid in titles]
                    path.append(f"{section_id} {title}".strip())
                    titles[section_id] = title
                    current = Section(doc, section_id, title, path, page.page, page.page)
                    expected_tz, buffer_tz, buffer_page = 1, None, page.page
                    title_pending = not title
                    continue
                tz = _TEXTZIFFER_RE.match(stripped)
                if title_pending and not tz:
                    # The heading number stood on a line of its own; the title follows.
                    current.title = titles[current.section_id] = stripped
                    current.path[-1] = f"{current.section_id} {stripped}"
                    title_pending = False
                    continue
                title_pending = False
                if tz and int(tz.group("tz")) == expected_tz:
                    flush()
                    buffer_tz, buffer_page = expected_tz, page.page
                    expected_tz += 1
                # The Textziffer number stays in the text, as in the source document.
                buffer.append(stripped)
                buffer_last_page = current.page_end = page.page
        flush()
        sections.append(current)
    # A heading immediately followed by a sub-heading ("AT 4" then "AT 4.1") yields an
    # empty section. Its title survives in the children's heading path.
    return [s for s in sections if s.paragraphs]


# --- Splitting helpers ----------------------------------------------------

def split_windows(text: str, size: int, overlap: int) -> list[str]:
    """Character windows of about `size`, ending at a sentence or word boundary.

    The cut is moved back to the last sentence end in the final third of the window,
    else to the last space, so that a window never ends inside a word.
    """
    if len(text) <= size:
        return [text]
    windows, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            floor = start + (2 * size) // 3
            cut = max(text.rfind(". ", floor, end), text.rfind(".\n", floor, end))
            if cut != -1:
                end = cut + 1
            else:
                space = text.rfind(" ", floor, end)
                end = space if space != -1 else end
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        # Start the next window at a word boundary.
        space = text.find(" ", next_start, end)
        start = space + 1 if space != -1 else next_start
    return [w for w in windows if w]


# --- Strategies -----------------------------------------------------------

def chunk_structure(sections: list[Section], size: int = config.CHUNK_SIZE,
                    overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    min_len = int(config.MIN_CHUNK_SHARE * size)
    for section in sections:
        groups: list[list[tuple[int | None, str, int, int]]] = []
        current: list[tuple[int | None, str, int, int]] = []
        length = 0
        for para in section.paragraphs:
            para_len = len(para[1])
            if para_len > size:
                if current:
                    groups.append(current)
                    current, length = [], 0
                for window in split_windows(para[1], size, overlap):
                    groups.append([(para[0], window, para[2], para[3])])
                continue
            if current and length + para_len + 1 > size:
                groups.append(current)
                current, length = [], 0
            current.append(para)
            length += para_len + 1
        if current:
            # A short tail is merged backwards rather than left as a near-empty chunk.
            tail_len = sum(len(p[1]) for p in current)
            if groups and tail_len < min_len and sum(len(p[1]) for p in groups[-1]) + tail_len <= size + min_len:
                groups[-1].extend(current)
            else:
                groups.append(current)
        for group in groups:
            tz = sorted({p[0] for p in group if p[0] is not None})
            chunks.append(Chunk(
                chunk_id="",
                doc=section.doc,
                section_id=section.section_id,
                section_title=section.title,
                heading_path=list(section.path),
                page_start=min(p[2] for p in group),
                page_end=max(p[3] for p in group),
                textziffern=tz,
                text="\n".join(p[1] for p in group),
            ))
    return _number(chunks)


def chunk_fixed(pages: list[Page], size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Character windows over each document's full text, blind to structure."""
    chunks: list[Chunk] = []
    for doc in dict.fromkeys(p.doc for p in pages):
        doc_pages = [p for p in pages if p.doc == doc]
        # Offsets of page starts and section headings in the concatenated text, so that
        # each window can be labelled afterwards.
        full, page_starts, headings = "", [], []
        for page in doc_pages:
            page_starts.append((len(full), page.page))
            for line in page.text.split("\n"):
                h = parse_heading(line.strip())
                if h:
                    headings.append((len(full), h[0], h[1]))
                full += line + "\n"
        search_from = 0
        for window in split_windows(full, size, overlap):
            offset = full.find(window, search_from)
            offset = offset if offset != -1 else search_from
            search_from = offset + 1
            section = [h for h in headings if h[0] <= offset]
            section_id, title = (section[-1][1], section[-1][2]) if section else ("PREAMBLE", "")
            pages_in = [pg for start, pg in page_starts if start <= offset + len(window)]
            first = [pg for start, pg in page_starts if start <= offset]
            chunks.append(Chunk(
                chunk_id="",
                doc=doc,
                section_id=section_id,
                section_title=title,
                heading_path=[],     # the baseline gets no heading context
                page_start=first[-1] if first else doc_pages[0].page,
                page_end=pages_in[-1] if pages_in else doc_pages[-1].page,
                textziffern=[],
                text=window,
            ))
    return _number(chunks)


def _number(chunks: list[Chunk]) -> list[Chunk]:
    counters: dict[str, int] = {}
    for c in chunks:
        stem = c.doc.rsplit(".", 1)[0]
        counters[stem] = counters.get(stem, 0) + 1
        c.chunk_id = f"{stem}:{counters[stem]:04d}"
    return chunks


def build_chunks(pages: list[Page], strategy: str = config.CHUNK_STRATEGY,
                 size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    if not 0 <= overlap < size // 2:
        raise ValueError(f"overlap must be below half the chunk size, got {overlap} for size {size}")
    if strategy == "structure":
        return chunk_structure(parse_sections(pages), size, overlap)
    if strategy == "fixed":
        return chunk_fixed(pages, size, overlap)
    raise ValueError(f"unknown chunk strategy {strategy!r}")
