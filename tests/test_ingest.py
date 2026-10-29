from regrag import config
from regrag.ingest import columns_to_text, find_column_split, find_furniture, is_toc_page, strip_furniture


def _word(text, x0, top, width=None):
    width = width if width is not None else 4.0 * len(text)
    return {"text": text, "x0": x0, "x1": x0 + width, "top": top}


def _two_column_page():
    """Left column x 50-330, right column x 360-560, page width 600."""
    words = [_word("AT", 50, 100), _word("4.4.2", 65, 100), _word("Compliance-Funktion", 100, 100)]
    rows = [
        (120, "1 Jedes Institut muss über eine Compliance-Funktion verfügen", "Die Funktion kann angebunden"),
        (132, "um Risiken aus der Nichteinhaltung entgegenzuwirken", "werden an andere Einheiten"),
        (150, "2 Die Compliance-Funktion hat hinzuwirken auf Verfahren", "Wesentlich sind Regelungen"),
        (162, "und auf entsprechende Kontrollen im Institut selbst", "mit hohem Risiko für Kunden"),
    ]
    for top, left, right in rows:
        x = 50
        for token in left.split():
            words.append(_word(token, x, top))
            x += 4.0 * len(token) + 4
        x = 360
        for token in right.split():
            words.append(_word(token, x, top))
            x += 4.0 * len(token) + 4
    return words


def test_column_split_found_in_whitespace_gap():
    words = _two_column_page()
    split = find_column_split(words, page_width=600)
    assert split is not None
    left_max = max(w["x1"] for w in words if w["x0"] < 340)
    assert left_max <= split <= 360


def test_drawn_vertical_rule_wins_over_whitespace():
    assert find_column_split(_two_column_page(), 600, rules=[345.0]) == 345.0


def test_single_column_page_has_no_split():
    words = []
    for top in range(100, 400, 12):
        x = 50
        for token in "Die Geschäftsleitung hat eine nachhaltige Geschäftsstrategie festzulegen und zu überprüfen".split():
            words.append(_word(token, x, top))
            x += 4.0 * len(token) + 4
    assert find_column_split(words, 600) is None


def test_explanations_attached_to_their_textziffer():
    text = columns_to_text(_two_column_page(), split=345)
    lines = text.split("\n")
    assert lines[0].startswith("AT 4.4.2")
    tz1 = next(i for i, ln in enumerate(lines) if ln.startswith("1 Jedes"))
    expl1 = next(i for i, ln in enumerate(lines) if "Die Funktion kann angebunden" in ln)
    tz2 = next(i for i, ln in enumerate(lines) if ln.startswith("2 Die Compliance"))
    expl2 = next(i for i, ln in enumerate(lines) if "Wesentlich sind Regelungen" in ln)
    assert tz1 < expl1 < tz2 < expl2
    assert lines[expl1].startswith(config.EXPLANATION_PREFIX)
    # No line mixes the two columns.
    assert not any("Jedes" in ln and "Funktion kann" in ln for ln in lines)


def test_running_header_and_page_numbers_removed_but_body_kept():
    topics = ["Strategien", "Stresstests", "Personal", "Auslagerung", "Dokumentation"]
    pages = [f"Mindestanforderungen an das Risikomanagement\nAT {i}.1 Titel\n1 Anforderungen zu {t}\nSeite {i} von 9"
             for i, t in enumerate(topics, start=1)]
    furniture = find_furniture(pages)
    cleaned = strip_furniture(pages, furniture)
    for i, (text, topic) in enumerate(zip(cleaned, topics), start=1):
        assert "Mindestanforderungen" not in text
        assert "Seite" not in text
        assert f"AT {i}.1 Titel" in text          # headings are never furniture
        assert f"1 Anforderungen zu {topic}" in text


def test_toc_page_detected():
    toc = "Inhaltsverzeichnis\n" + "\n".join(f"AT {i}.1 Abschnitt {i} ........ {i + 3}" for i in range(1, 8))
    body = "AT 4.1 Risikotragfähigkeit\n1 Auf der Grundlage des Gesamtrisikoprofils ist sicherzustellen."
    assert is_toc_page(toc)
    assert not is_toc_page(body)
