import pytest

from regrag import config
from regrag.chunk import build_chunks, chunk_structure, parse_heading, parse_sections, split_windows
from regrag.text import match_key


def test_heading_versus_citation():
    assert parse_heading("BTO 2.1 Funktionstrennung") == ("BTO 2.1", "Funktionstrennung")
    assert parse_heading("AT 4.4.2 Compliance-Funktion") == ("AT 4.4.2", "Compliance-Funktion")
    assert parse_heading("AT 4.3.2 Tz. 3 gilt entsprechend.") is None
    assert parse_heading("AT 4.4.1 und AT 4.4.2 sind zu beachten") is None
    assert parse_heading("BTR 3.2 zusätzliche Anforderungen") is None     # lower-case: not a title


def test_sections_and_parent_path(pages):
    sections = {s.section_id: s for s in parse_sections(pages)}
    assert set(sections) == {"PREAMBLE", "AT 4.1", "AT 4.4.2", "BTR 2.2", "BTR 2.3"}
    # "AT 4" has no body of its own, but survives as the parent heading.
    assert sections["AT 4.1"].path == ["AT 4 Allgemeine Anforderungen an das Risikomanagement",
                                       "AT 4.1 Risikotragfähigkeit"]


def test_textziffer_sequence_rejects_false_numbers(pages):
    s = {s.section_id: s for s in parse_sections(pages)}["AT 4.1"]
    # "3 Monate ..." is accepted because 3 is next in sequence -- the rule cannot tell, and
    # the test documents that limit. The cross-reference "gemäß Tz. 1" starts no paragraph.
    assert [p[0] for p in s.paragraphs] == [1, 2, 3]
    assert "gemäß Tz. 1" in s.paragraphs[2][1]


def test_running_header_repeat_does_not_split_section(pages):
    sections = [s for s in parse_sections(pages) if s.section_id == "AT 4.4.2"]
    assert len(sections) == 1
    s = sections[0]
    assert (s.page_start, s.page_end) == (1, 2)
    assert [p[0] for p in s.paragraphs] == [1, 2]
    # The citation line stays in the body of Tz. 2.
    assert "AT 4.3.2 Tz. 3 gilt entsprechend." in s.paragraphs[1][1]


def test_structure_chunks_never_cross_sections(pages):
    chunks = build_chunks(pages, "structure", size=1200, overlap=200)
    assert all(c.section_id != "BTR 2.2" or "Anlagebuches" not in c.text for c in chunks)
    btr = [c for c in chunks if c.section_id == "BTR 2.3"]
    assert len(btr) == 1 and btr[0].embed_text.startswith("BTR 2.3 Marktpreisrisiken des Anlagebuches\n")


def test_structure_chunks_pack_whole_textziffern(pages):
    sections = parse_sections(pages)
    paras = {(s.section_id, p[0]): p[1] for s in sections for p in s.paragraphs}
    for size in (200, 1200):
        for c in chunk_structure(sections, size=size, overlap=30):
            if c.section_id == "AT 4.1":
                # A chunk is exactly a run of complete Textziffern, never a cut through one.
                assert c.text == "\n".join(paras[("AT 4.1", tz)] for tz in c.textziffern)
    packed = [c for c in chunk_structure(sections, 1200, 200) if c.section_id == "AT 4.1"]
    assert len(packed) == 1 and packed[0].textziffern == [1, 2, 3]
    split = [c for c in chunk_structure(sections, 200, 30) if c.section_id == "AT 4.1"]
    assert len(split) > 1


def test_fixed_chunks_cover_all_text_with_overlap(pages):
    chunks = build_chunks(pages, "fixed", size=300, overlap=60)
    full = match_key("\n".join(p.text for p in pages))
    keys = [match_key(c.text) for c in chunks]
    assert full.startswith(keys[0]) and full.endswith(keys[-1])
    # Each window starts inside the previous one: no text falls between two chunks.
    for a, b in zip(keys, keys[1:]):
        assert b[:20] in a
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.heading_path == [] for c in chunks)


def test_split_windows_end_on_boundaries():
    text = " ".join(f"Satz {i} endet hier." for i in range(100))
    windows = split_windows(text, size=200, overlap=40)
    assert all(len(w) <= 200 for w in windows)
    assert all(w.endswith(".") for w in windows)
    # Overlap: the start of each window appeared in the previous one.
    for a, b in zip(windows, windows[1:]):
        assert b[:15] in a


def test_bad_overlap_rejected(pages):
    with pytest.raises(ValueError):
        build_chunks(pages, "fixed", size=200, overlap=150)


def test_chunk_ids_unique_and_stable(pages):
    a = build_chunks(pages, config.CHUNK_STRATEGY)
    b = build_chunks(pages, config.CHUNK_STRATEGY)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
