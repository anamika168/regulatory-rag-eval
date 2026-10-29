"""Evaluation: labelled questions, span-based relevance, retrieval / abstention /
groundedness metrics.

Relevance is decided by text, never by chunk ids or section labels: a chunk is relevant
to a question if it contains (most of) a gold span from data/eval/questions.yaml. That
makes the labels independent of the chunker, so the same question set can score the
structure-aware chunker, the fixed-size baseline and every chunk size in the ablation.
Labelling chunk ids instead would have to be redone for every configuration and would
quietly favour the one it was made with.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from regrag import config
from regrag.chunk import Chunk
from regrag.generate import Answer
from regrag.retrieve import Hit
from regrag.text import match_key, span_coverage


# --- Questions ------------------------------------------------------------

@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    source_doc: str | None = None
    gold_section: str | None = None
    gold_spans: list[str] = field(default_factory=list)
    note: str = ""


def load_questions(path: Path = config.QUESTIONS_PATH) -> list[Question]:
    """Load and validate the question set; raises ValueError listing every problem."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    questions, problems, seen = [], [], set()
    for i, entry in enumerate(raw):
        qid = entry.get("id", f"<entry {i}>")
        for key in ("id", "question", "answerable"):
            if key not in entry:
                problems.append(f"{qid}: missing field {key!r}")
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        spans = entry.get("gold_span")
        spans = [spans] if isinstance(spans, str) else list(spans or [])
        if entry.get("answerable") is True and not spans:
            problems.append(f"{qid}: answerable but no gold_span")
        if entry.get("answerable") is False and (spans or entry.get("source_doc")):
            problems.append(f"{qid}: unanswerable questions take no gold_span or source_doc")
        if any(len(match_key(s)) < 20 for s in spans):
            problems.append(f"{qid}: gold_span too short to be unambiguous (< 20 letters)")
        questions.append(Question(
            id=qid, question=entry.get("question", ""), answerable=bool(entry.get("answerable")),
            source_doc=entry.get("source_doc"), gold_section=entry.get("gold_section"),
            gold_spans=spans, note=entry.get("note", "") or "",
        ))
    if problems:
        raise ValueError("questions.yaml has problems:\n  " + "\n  ".join(problems))
    return questions


# --- Relevance ------------------------------------------------------------

class RelevanceJudge:
    """Which chunks contain which gold span. Chunk match keys are computed once."""

    def __init__(self, chunks: list[Chunk], min_coverage: float = config.SPAN_MIN_COVERAGE):
        self.chunks = chunks
        self.keys = [match_key(c.text) for c in chunks]
        self.min_coverage = min_coverage
        self._index = {c.chunk_id: i for i, c in enumerate(chunks)}

    def coverage(self, chunk: Chunk, span: str) -> float:
        i = self._index.get(chunk.chunk_id)
        key = self.keys[i] if i is not None else match_key(chunk.text)
        return span_coverage(key, match_key(span))

    def relevant_chunks(self, span: str, source_doc: str | None = None) -> set[str]:
        """Ids of all chunks holding at least min_coverage of the span in one piece.

        Pre-filter: a contiguous match covering a share c of the span contains at least
        one of the span's aligned blocks of length c*n/2, so chunks containing none of
        those blocks can be skipped without computing the alignment.
        """
        span_key = match_key(span)
        n = len(span_key)
        block = max(1, int(self.min_coverage * n / 2))
        blocks = {span_key[i:i + block] for i in range(0, n - block + 1, block)}
        found = set()
        for chunk, key in zip(self.chunks, self.keys):
            if source_doc and chunk.doc != source_doc:
                continue
            if not any(b in key for b in blocks):
                continue
            if span_coverage(key, span_key) >= self.min_coverage:
                found.add(chunk.chunk_id)
        return found


# --- Retrieval metrics ----------------------------------------------------

def evaluate_retrieval_question(q: Question, hits: list[Hit], judge: RelevanceJudge,
                                ks: tuple[int, ...] = config.RECALL_AT_K) -> dict:
    """Per-question retrieval record. `hits` must be at least max(ks) deep."""
    span_sets = [judge.relevant_chunks(s, q.source_doc) for s in q.gold_spans]
    retrieved = []
    for h in hits:
        relevant = any(h.chunk.chunk_id in s for s in span_sets)
        retrieved.append({"rank": h.rank, "chunk_id": h.chunk.chunk_id, "doc": h.chunk.doc,
                          "section_id": h.chunk.section_id, "score": round(h.score, 4),
                          "dense_score": None if h.dense_score is None else round(h.dense_score, 4),
                          "relevant": relevant})
    dense_scores = [h.dense_score for h in hits if h.dense_score is not None]
    row = {
        "id": q.id, "question": q.question, "answerable": q.answerable,
        "gold_section": q.gold_section,
        "top_dense_score": round(max(dense_scores), 4) if dense_scores else None,
        "retrieved": retrieved,
    }
    if not q.answerable:
        return row

    first = next((r["rank"] for r in retrieved if r["relevant"]), None)
    row["n_gold_spans"] = len(span_sets)
    row["spans_in_index"] = sum(bool(s) for s in span_sets)
    row["gold_chunk_ids"] = sorted(set().union(*span_sets)) if span_sets else []
    row["first_relevant_rank"] = first
    row["reciprocal_rank"] = 1.0 / first if first else 0.0
    for k in ks:
        top = {r["chunk_id"] for r in retrieved[:k]}
        row[f"recall@{k}"] = sum(bool(s & top) for s in span_sets) / len(span_sets)
        if q.gold_section:
            row[f"section_hit@{k}"] = any(r["section_id"] == q.gold_section for r in retrieved[:k])
    return row


def summarize_retrieval(rows: list[dict], ks: tuple[int, ...] = config.RECALL_AT_K) -> dict:
    """Recall@k and MRR over answerable questions.

    Reported twice: over all answerable questions (end to end; a gold span lost at
    extraction or chunking counts as a miss) and over those whose gold spans all exist
    in the index (retrieval alone). The gap between the two is the ingest/chunking loss.
    """
    answerable = [r for r in rows if r["answerable"]]
    in_index = [r for r in answerable if r["spans_in_index"] == r["n_gold_spans"]]
    summary = {"n_questions": len(rows), "n_answerable": len(answerable),
               "n_gold_in_index": len(in_index)}
    for name, subset in (("all", answerable), ("in_index", in_index)):
        for k in ks:
            summary[f"{name}_recall@{k}"] = _mean(r[f"recall@{k}"] for r in subset)
        summary[f"{name}_mrr"] = _mean(r["reciprocal_rank"] for r in subset)
    with_section = [r for r in answerable if r.get("gold_section")]
    for k in ks:
        summary[f"section_hit@{k}"] = _mean(r[f"section_hit@{k}"] for r in with_section)
    return summary


def _mean(values) -> float | None:
    values = list(values)
    return round(float(np.mean(values)), 4) if values else None


# --- Abstention -----------------------------------------------------------

def threshold_sweep(rows: list[dict], thresholds=config.THRESHOLD_SWEEP, k: int = config.TOP_K) -> list[dict]:
    """What a retrieval-score threshold would do, for each candidate threshold.

    answered_answerable   -- share of answerable questions that pass (coverage)
    useful_answered       -- ... that pass AND have the gold passage in the top k
    abstained_unanswerable -- share of unanswerable questions stopped (the goal)

    Choosing the threshold from this table and reporting the same table is optimistic:
    the threshold is fitted on the questions it is evaluated on. With a few dozen
    questions there is no room for a held-out split, so the table is reported as a
    curve, not as a single tuned number.
    """
    answerable = [r for r in rows if r["answerable"] and r["top_dense_score"] is not None]
    unanswerable = [r for r in rows if not r["answerable"] and r["top_dense_score"] is not None]
    out = []
    for t in thresholds:
        out.append({
            "threshold": t,
            "answered_answerable": _mean(r["top_dense_score"] >= t for r in answerable),
            "useful_answered": _mean(r["top_dense_score"] >= t and r[f"recall@{k}"] > 0 for r in answerable),
            "abstained_unanswerable": _mean(r["top_dense_score"] < t for r in unanswerable),
        })
    return out


def separability(rows: list[dict]) -> dict:
    """How well the top dense score separates answerable from unanswerable questions.

    AUC = probability that a random answerable question scores higher than a random
    unanswerable one (0.5 = no separation, 1.0 = a perfect threshold exists). If this is
    near 0.5, no threshold can work and abstention has to come from the generator.
    """
    pos = [r["top_dense_score"] for r in rows if r["answerable"] and r["top_dense_score"] is not None]
    neg = [r["top_dense_score"] for r in rows if not r["answerable"] and r["top_dense_score"] is not None]
    if not pos or not neg:
        return {"auc": None}
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return {
        "auc": round(wins / (len(pos) * len(neg)), 4),
        "answerable_score": _describe(pos),
        "unanswerable_score": _describe(neg),
    }


def _describe(values: list[float]) -> dict:
    a = np.array(values)
    return {"min": round(float(a.min()), 3), "median": round(float(np.median(a)), 3),
            "max": round(float(a.max()), 3)}


# --- Generation metrics ---------------------------------------------------

def evaluate_answer(q: Question, answer: Answer, hits: list[Hit], judge: RelevanceJudge) -> dict:
    """Per-question generation record: abstention, citation validity, quote grounding."""
    row = {
        "id": q.id, "question": q.question, "answerable_gold": q.answerable,
        "abstained": answer.abstained, "abstained_by": answer.abstained_by,
        "error": answer.error, "model": answer.model,
        "answer": answer.answer, "citations": answer.citations, "quotes": answer.quotes,
        "passage_ids": answer.passage_ids,
        "input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens,
    }
    gold_sets = [judge.relevant_chunks(s, q.source_doc) for s in q.gold_spans] if q.answerable else []
    gold_ids = set().union(*gold_sets) if gold_sets else set()
    row["retrieved_gold"] = any(h.chunk.chunk_id in gold_ids for h in hits)

    if answer.abstained or answer.error:
        return row

    n = len(hits)
    row["citations_valid"] = all(1 <= c <= n for c in answer.citations) and bool(answer.citations)
    grounded = []
    for quote in answer.quotes:
        p = quote["passage"]
        cov = judge.coverage(hits[p - 1].chunk, quote["text"]) if 1 <= p <= n else 0.0
        grounded.append(round(cov, 3))
    row["quote_coverage"] = grounded
    row["n_quotes"] = len(grounded)
    row["n_grounded_quotes"] = sum(c >= config.QUOTE_MIN_COVERAGE for c in grounded)
    quoted_passages = {q_["passage"] for q_ in answer.quotes}
    row["fully_grounded"] = (bool(grounded) and row["n_grounded_quotes"] == len(grounded)
                             and set(answer.citations) <= quoted_passages)
    row["cites_gold"] = any(1 <= c <= n and hits[c - 1].chunk.chunk_id in gold_ids for c in answer.citations)
    return row


def summarize_generation(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable_gold"]]
    unanswerable = [r for r in rows if not r["answerable_gold"]]
    answered = [r for r in answerable if not r["abstained"] and not r["error"]]
    quotes = sum(r.get("n_quotes", 0) for r in rows)
    grounded_quotes = sum(r.get("n_grounded_quotes", 0) for r in rows)
    gold_available = [r for r in answered if r["retrieved_gold"]]
    summary = {
        "n_questions": len(rows),
        "n_errors": sum(bool(r["error"]) for r in rows),
        # Answerable questions
        "n_answerable": len(answerable),
        "false_abstention_rate": _mean(r["abstained"] for r in answerable),
        "false_abstention_by_retrieval": sum(r["abstained_by"] == "retrieval" for r in answerable),
        "false_abstention_by_model": sum(r["abstained_by"] == "model" for r in answerable),
        "citations_valid_rate": _mean(r["citations_valid"] for r in answered),
        "fully_grounded_rate": _mean(r["fully_grounded"] for r in answered),
        "quote_precision": round(grounded_quotes / quotes, 4) if quotes else None,
        # Of the answers where the gold passage was among the passages shown, how often
        # the answer actually cites it.
        "cites_gold_rate": _mean(r["cites_gold"] for r in gold_available),
        # Unanswerable questions
        "n_unanswerable": len(unanswerable),
        "correct_abstention_rate": _mean(r["abstained"] for r in unanswerable),
        "abstained_by_retrieval": sum(r["abstained_by"] == "retrieval" for r in unanswerable),
        "abstained_by_model": sum(r["abstained_by"] == "model" for r in unanswerable),
        "answered_unanswerable": sum(not r["abstained"] and not r["error"] for r in unanswerable),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }
    # Over every answered question, answerable or not: an answer to an unanswerable
    # question can still be faithful to the passages it was shown.
    verdicts = [r["judge"]["verdict"] for r in rows if r.get("judge") and r["judge"].get("verdict")]
    if verdicts:
        for v in ("supported", "partially_supported", "unsupported"):
            summary[f"judge_{v}"] = _mean(x == v for x in verdicts)
    return summary
