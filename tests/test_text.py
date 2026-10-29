from regrag.text import clean_text, key_with_offsets, match_key, span_coverage, tokenize


def test_line_break_hyphenation_joined_only_before_lower_case():
    assert clean_text("Risiko-\ncontrolling") == "Risikocontrolling"
    assert clean_text("Compliance-\nFunktion") == "Compliance-\nFunktion"
    assert clean_text("Risikosteuerungs- und -controllingprozesse") == "Risikosteuerungs- und -controllingprozesse"


def test_clean_text_removes_soft_hyphens_and_collapses_spaces():
    assert clean_text("Ge­schäfts   leitung  \n  x") == "Geschäfts leitung\nx"


def test_match_key_ignores_whitespace_hyphens_and_quote_styles():
    assert match_key("„Risiko-\ncontrolling“") == match_key('"Risikocontrolling"') == "risikocontrolling"


def test_span_coverage_full_partial_and_none():
    passage = match_key("Die Positionen des Handelsbuches sind täglich zu bewerten.")
    assert span_coverage(passage, match_key("sind täglich zu bewerten")) == 1.0
    partial = span_coverage(passage, match_key("sind täglich zu bewerten und zu melden"))
    assert 0.5 < partial < 1.0
    assert span_coverage(passage, match_key("")) == 0.0


def test_key_with_offsets_points_back_into_original_text():
    text = "Die Compliance-Funktion"
    key, offsets = key_with_offsets(text)
    assert key == match_key(text)
    start = key.index("compliance")
    assert text[offsets[start]:offsets[start + len("compliancefunktion") - 1] + 1] == "Compliance-Funktion"


def test_tokenize_keeps_modal_verbs():
    tokens = tokenize("Das Institut muss die Risiken begrenzen und sollte sie überwachen.")
    assert "muss" in tokens and "sollte" in tokens
    assert "das" not in tokens and "die" not in tokens
