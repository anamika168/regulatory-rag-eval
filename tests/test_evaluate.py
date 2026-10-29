import textwrap

import pytest

from regrag import config
from regrag.chunk import Chunk, build_chunks
from regrag.evaluate import (Question, RelevanceJudge, evaluate_retrieval_question, load_questions, separability,
                             summarize_retrieval, threshold_sweep)
from regrag.retrieve import Hit, Retriever


def _chunk(cid, text, section="AT 1"):
    return Chunk(chunk_id=cid, doc="d.pdf", section_id=section, section_title="", heading_path=[],
                 page_start=1, page_end=1, textziffern=[], text=text)


def test_relevance_is_text_based_and_tolerates_boundary_cuts():
    span = "Die mit Marktpreisrisiken behafteten Positionen des Handelsbuches sind täglich zu bewerten."
    chunks = [
        _chunk("a", "Vorher. " + span + " Nachher."),
        _chunk("b", "Die mit Marktpreisrisiken behafteten Positionen des Handelsbuches"),   # ~70 % of the span
        _chunk("c", "sind täglich zu bewerten."),                                           # ~25 %
        _chunk("d", "Etwas ganz anderes über das Anlagebuch."),
    ]
    judge = RelevanceJudge(chunks, min_coverage=0.6)
    assert judge.relevant_chunks(span) == {"a", "b"}


def test_source_doc_restricts_relevance():
    span = "Jedes Institut muss über eine funktionsfähige Interne Revision verfügen."
    a, b = _chunk("a", span), _chunk("b", span)
    b.doc = "other.pdf"
    assert RelevanceJudge([a, b]).relevant_chunks(span, source_doc="other.pdf") == {"b"}


def test_recall_and_mrr_on_known_ranking():
    chunks = [_chunk(str(i), f"Text Nummer {i} mit genug Inhalt für einen Abschnitt.") for i in range(10)]
    chunks[3].text = "Hier steht die Antwort: das Anlagebuch ist mindestens vierteljährlich zu bewerten."
    judge = RelevanceJudge(chunks)
    q = Question(id="q", question="?", answerable=True,
                 gold_spans=["das Anlagebuch ist mindestens vierteljährlich zu bewerten"], gold_section="AT 1")
    hits = [Hit(chunks[i], rank=r, score=1.0 - r / 10, dense_score=0.9 - r / 20)
            for r, i in enumerate([0, 1, 3, 2, 4, 5, 6, 7, 8, 9], start=1)]
    row = evaluate_retrieval_question(q, hits, judge)
    assert row["first_relevant_rank"] == 3
    assert row["reciprocal_rank"] == pytest.approx(1 / 3)
    assert (row["recall@1"], row["recall@3"], row["recall@5"]) == (0.0, 1.0, 1.0)
    assert row["top_dense_score"] == pytest.approx(0.85)


def test_multi_span_recall_is_fractional():
    chunks = [_chunk("h", "Positionen des Handelsbuches sind täglich zu bewerten."),
              _chunk("x", "Unbeteiligter Text ohne Bezug zur Frage hier."),
              _chunk("a", "Positionen des Anlagebuches sind mindestens vierteljährlich zu bewerten.")]
    judge = RelevanceJudge(chunks)
    q = Question(id="q", question="?", answerable=True,
                 gold_spans=["Positionen des Handelsbuches sind täglich zu bewerten",
                             "Positionen des Anlagebuches sind mindestens vierteljährlich zu bewerten"])
    hits = [Hit(c, rank=r, score=0.0, dense_score=None) for r, c in enumerate(chunks, start=1)]
    row = evaluate_retrieval_question(q, hits, judge, ks=(1, 3))
    assert row["recall@1"] == 0.5 and row["recall@3"] == 1.0


def test_summary_separates_extraction_loss_from_retrieval_loss(pages, embedder):
    chunks = build_chunks(pages, "structure")
    retriever = Retriever(chunks, method="dense", embedder=embedder)
    questions = [
        Question("in", "Wie oft sind Positionen des Handelsbuches zu bewerten?", True,
                 gold_spans=["Positionen des Handelsbuches sind täglich zu bewerten"]),
        Question("lost", "Was steht in der verlorenen Fußnote?", True,
                 gold_spans=["Dieser Satz kommt im Korpus nirgends vor und kann nicht gefunden werden"]),
        Question("none", "Wie hoch ist die Einlagensicherung?", False),
    ]
    judge = RelevanceJudge(chunks)
    rows = [evaluate_retrieval_question(q, h, judge) for q, h in
            zip(questions, retriever.search_many([q.question for q in questions], 10))]
    s = summarize_retrieval(rows)
    assert s["n_answerable"] == 2 and s["n_gold_in_index"] == 1
    assert s["in_index_recall@1"] == 1.0 and s["all_recall@1"] == 0.5


def test_threshold_sweep_and_separability():
    rows = [{"answerable": True, "top_dense_score": 0.8, f"recall@{config.TOP_K}": 1.0},
            {"answerable": True, "top_dense_score": 0.5, f"recall@{config.TOP_K}": 0.0},
            {"answerable": False, "top_dense_score": 0.4},
            {"answerable": False, "top_dense_score": 0.6}]
    sweep = {r["threshold"]: r for r in threshold_sweep(rows, thresholds=(0.45, 0.7))}
    assert sweep[0.45]["answered_answerable"] == 1.0
    assert sweep[0.45]["useful_answered"] == 0.5
    assert sweep[0.45]["abstained_unanswerable"] == 0.5
    assert sweep[0.7]["abstained_unanswerable"] == 1.0
    assert separability(rows)["auc"] == 0.75


def test_question_file_validation(tmp_path):
    bad = tmp_path / "q.yaml"
    bad.write_text(textwrap.dedent("""
        - id: a
          question: x
          answerable: true
        - id: a
          question: y
          answerable: false
          gold_span: "darf es nicht geben bei unbeantwortbaren Fragen"
    """), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_questions(bad)
    message = str(exc.value)
    assert "answerable but no gold_span" in message
    assert "duplicate id" in message
    assert "unanswerable questions take no gold_span" in message


def test_repository_question_set_is_valid():
    questions = load_questions(config.QUESTIONS_PATH)
    assert questions, "the labelled question set is empty"
    assert any(not q.answerable for q in questions), "abstention cannot be measured without unanswerable questions"
