import json
from types import SimpleNamespace

import pytest

from regrag.chunk import build_chunks
from regrag.evaluate import Question, RelevanceJudge, evaluate_answer, summarize_generation
from regrag.generate import ANSWER_SCHEMA, Generator, build_prompt
from regrag.pipeline import answer_question
from regrag.retrieve import Retriever


class FakeMessages:
    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.calls = payload, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="thinking", thinking=""),
                   SimpleNamespace(type="text", text=json.dumps(self.payload, ensure_ascii=False))]
        return SimpleNamespace(content=content, stop_reason=self.stop_reason, model=kwargs["model"],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))


def _client(payload, stop_reason="end_turn"):
    return SimpleNamespace(messages=FakeMessages(payload, stop_reason))


@pytest.fixture
def retriever(pages, embedder):
    return Retriever(build_chunks(pages, "structure"), method="dense", embedder=embedder)


QUESTION = "Wie oft sind die Positionen des Handelsbuches zu bewerten?"


def test_prompt_numbers_passages_with_source(retriever):
    hits = retriever.search(QUESTION, k=2)
    prompt = build_prompt(QUESTION, hits)
    assert "[1] (marisk_test.pdf, BTR 2.2 Marktpreisrisiken des Handelsbuches, S. 2)" in prompt
    assert "[2]" in prompt and prompt.endswith(f"Frage: {QUESTION}")


def test_request_uses_structured_output_and_no_sampling_params(retriever):
    client = _client({"answerable": True, "answer": "täglich", "citations": [1], "quotes": []})
    Generator(model="m", client=client).answer(QUESTION, retriever.search(QUESTION, 2))
    call = client.messages.calls[0]
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": ANSWER_SCHEMA}
    assert "effort" not in call["output_config"]
    assert not {"temperature", "top_p", "top_k", "tool_choice"} & set(call)


def test_grounded_answer_scores_as_grounded(retriever):
    hits = retriever.search(QUESTION, k=3)
    quote = "Die mit Marktpreisrisiken behafteten Positionen des Handelsbuches sind täglich zu bewerten."
    client = _client({"answerable": True, "answer": "Täglich.", "citations": [1], "quotes": [{"passage": 1, "text": quote}]})
    answer, hits = answer_question(QUESTION, retriever, Generator(model="m", client=client), k=3, threshold=0.1, hits=hits)
    q = Question("q", QUESTION, True, gold_spans=[quote])
    row = evaluate_answer(q, answer, hits, RelevanceJudge(retriever.chunks))
    assert row["fully_grounded"] and row["cites_gold"] and row["citations_valid"]
    assert row["quote_coverage"] == [1.0]


def test_paraphrased_quote_is_not_grounded(retriever):
    hits = retriever.search(QUESTION, k=3)
    client = _client({"answerable": True, "answer": "Täglich.", "citations": [1],
                      "quotes": [{"passage": 1, "text": "Handelsbuchpositionen müssen jeden Tag neu bewertet werden."}]})
    answer, hits = answer_question(QUESTION, retriever, Generator(model="m", client=client), k=3, threshold=0.1, hits=hits)
    row = evaluate_answer(Question("q", QUESTION, True, gold_spans=["sind täglich zu bewerten bei Handelsbuch"]),
                          answer, hits, RelevanceJudge(retriever.chunks))
    assert not row["fully_grounded"] and row["n_grounded_quotes"] == 0


def test_retrieval_abstention_makes_no_api_call(retriever):
    client = _client({})
    answer, _ = answer_question("Kryptoverwahrung nach MiCAR", retriever, Generator(model="m", client=client),
                                threshold=0.99)
    assert answer.abstained_by == "retrieval"
    assert client.messages.calls == []


def test_model_abstention_and_errors_recorded(retriever):
    hits = retriever.search(QUESTION, 2)
    no = Generator(model="m", client=_client({"answerable": False, "answer": "Nicht enthalten.",
                                              "citations": [], "quotes": []})).answer("Einlagensicherung?", hits)
    assert no.abstained_by == "model"
    cut = Generator(model="m", client=_client({}, stop_reason="max_tokens")).answer(QUESTION, hits)
    assert cut.error and cut.answerable is None


def test_generation_summary_counts():
    base = {"error": None, "model": "m", "input_tokens": 1, "output_tokens": 1, "answer": "", "quotes": [],
            "citations": [], "passage_ids": []}
    rows = [
        dict(base, id="1", question="", answerable_gold=True, abstained=False, abstained_by=None, retrieved_gold=True,
             citations_valid=True, fully_grounded=True, n_quotes=1, n_grounded_quotes=1, cites_gold=True),
        dict(base, id="2", question="", answerable_gold=True, abstained=True, abstained_by="model", retrieved_gold=False),
        dict(base, id="3", question="", answerable_gold=False, abstained=True, abstained_by="retrieval", retrieved_gold=False),
        dict(base, id="4", question="", answerable_gold=False, abstained=False, abstained_by=None, retrieved_gold=False,
             citations_valid=True, fully_grounded=False, n_quotes=2, n_grounded_quotes=1, cites_gold=False),
    ]
    s = summarize_generation(rows)
    assert s["false_abstention_rate"] == 0.5 and s["false_abstention_by_model"] == 1
    assert s["correct_abstention_rate"] == 0.5 and s["answered_unanswerable"] == 1
    assert s["quote_precision"] == pytest.approx(2 / 3, abs=1e-4)
