"""Text normalisation shared by ingest, BM25 and evaluation.

Two different normalisations live here on purpose:

* clean_text() keeps the text readable. It is applied once at ingest, and what it
  produces is what gets embedded, shown to the generator and quoted back.
* match_key() destroys readability to make string comparison robust. It is used only to
  decide whether a gold span or a quote occurs in a chunk. PDF extraction varies in
  whitespace, hyphenation and quote characters between documents and even between pages,
  so comparing raw strings would count extraction noise as retrieval failure.
"""

import re
import unicodedata
from difflib import SequenceMatcher

_SOFT_HYPHEN = "­"

# A hyphen at the end of a line followed by a lower-case letter is a word broken across
# lines ("Risiko-\ncontrolling"). Followed by an upper-case letter it is usually a real
# compound hyphen ("Compliance-\nFunktion") and is kept. "Risikosteuerungs- und
# -controllingprozesse" is untouched because the hyphen is not at a line end.
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n(?=[a-zäöüß])")

_QUOTES = str.maketrans({"„": '"', "“": '"', "”": '"', "‚": "'", "‘": "'", "’": "'",
                         "–": "-", "—": "-", " ": " "})


def clean_text(text: str) -> str:
    """Readable normalisation: unicode forms, soft hyphens, line-break hyphenation."""
    text = unicodedata.normalize("NFKC", text).replace(_SOFT_HYPHEN, "")
    text = _LINE_BREAK_HYPHEN.sub(r"\1", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def match_key(text: str) -> str:
    """Lower-case letters and digits only.

    Removing every space and punctuation mark looks drastic, but it is what makes a
    verbatim quote match across "Risiko-controlling" / "Risikocontrolling" /
    "Risiko- controlling". For spans of a sentence or more, the chance that removing
    separators creates a false match is negligible.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).lower()
    return re.sub(r"[^0-9a-zäöüß]+", "", text)


def key_with_offsets(text: str) -> tuple[str, list[int]]:
    """match_key(text) plus, for every key character, its offset in `text`.

    Used to show the original wording around a fuzzy match, so that a gold span that
    does not match can be corrected by copying the real text. Assumes `text` has been
    through clean_text() (NFKC is idempotent, so offsets are stable).
    """
    key, offsets = [], []
    for i, ch in enumerate(text):
        ch = ch.translate(_QUOTES).lower()
        if len(ch) == 1 and re.match(r"[0-9a-zäöüß]", ch):
            key.append(ch)
            offsets.append(i)
    return "".join(key), offsets


def span_coverage(passage_key: str, span_key: str) -> float:
    """Share of the span that occurs in the passage as one contiguous piece.

    Both arguments must already be match keys. 1.0 means the span is contained;
    0.6 means the passage holds 60 % of it in one piece, e.g. because a chunk boundary
    fell inside the span.
    """
    if not span_key:
        return 0.0
    if span_key in passage_key:
        return 1.0
    matcher = SequenceMatcher(None, passage_key, span_key, autojunk=False)
    match = matcher.find_longest_match(0, len(passage_key), 0, len(span_key))
    return match.size / len(span_key)


# Short list on purpose: function words that carry no topic. Legal modal verbs such as
# "muss" and "soll" are NOT stopwords here -- in MaRisk the difference between "muss" and
# "sollte" is the difference between a requirement and a recommendation.
GERMAN_STOPWORDS = frozenset("""
der die das den dem des ein eine einer eines einem einen und oder aber auch als am an
auf aus bei bis durch für im in ist sind mit nach von vor zu zum zur über unter um wie
wird werden wurde es sich so dass daß nicht nur noch sowie bzw ggf etc this that the of
and or to in for is are be on with as by
""".split())


def tokenize(text: str) -> list[str]:
    """Tokens for BM25: lower-cased words, stopwords and single characters removed."""
    text = unicodedata.normalize("NFKC", text).lower().replace(_SOFT_HYPHEN, "")
    return [t for t in re.findall(r"\w+", text) if len(t) > 1 and t not in GERMAN_STOPWORDS]
