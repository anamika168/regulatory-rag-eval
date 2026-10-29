"""Shared fixtures: a synthetic MaRisk-like corpus and an embedder that needs no model.

The synthetic text imitates the structure that matters to the pipeline -- section
headings, Textziffern, cross-references, an Erläuterung -- not the regulation itself.
"""

import hashlib

import numpy as np
import pytest

from regrag.ingest import Page
from regrag.text import tokenize

PAGE_1 = """\
Rundschreiben zu den Mindestanforderungen an das Risikomanagement
AT 4 Allgemeine Anforderungen an das Risikomanagement
AT 4.1 Risikotragfähigkeit
1 Auf der Grundlage des Gesamtrisikoprofils ist sicherzustellen, dass die wesentlichen Risiken
laufend abgedeckt sind.
2 Das Institut hat einen internen Prozess zur Sicherstellung der Risikotragfähigkeit einzurichten.
3 Monate nach Beginn des Geschäftsjahres ist die Planung zu überprüfen; gemäß Tz. 1 gilt dies
entsprechend.
AT 4.4.2 Compliance-Funktion
1 Jedes Institut muss über eine Compliance-Funktion verfügen, um den Risiken, die sich aus der
Nichteinhaltung rechtlicher Regelungen und Vorgaben ergeben können, entgegenzuwirken.
Erläuterung: Die Compliance-Funktion kann an andere Kontrolleinheiten angebunden werden."""

PAGE_2 = """\
AT 4.4.2 Compliance-Funktion
2 Die Compliance-Funktion hat auf die Implementierung wirksamer Verfahren hinzuwirken.
AT 4.3.2 Tz. 3 gilt entsprechend.
BTR 2.2 Marktpreisrisiken des Handelsbuches
1 Die mit Marktpreisrisiken behafteten Positionen des Handelsbuches sind täglich zu bewerten.
BTR 2.3 Marktpreisrisiken des Anlagebuches
1 Die mit Marktpreisrisiken behafteten Positionen des Anlagebuches sind mindestens
vierteljährlich zu bewerten."""


@pytest.fixture
def pages() -> list[Page]:
    return [Page(doc="marisk_test.pdf", page=1, text=PAGE_1), Page(doc="marisk_test.pdf", page=2, text=PAGE_2)]


class HashingEmbedder:
    """Bag-of-words hashed into a fixed dimension. Lexical, so it tests the plumbing of
    dense retrieval (normalisation, FAISS, score handling), not semantic quality."""

    name = "test/hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def encode_passages(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
