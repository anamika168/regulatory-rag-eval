#!/usr/bin/env python3
"""
Ask one question and see every step: retrieved passages with scores, the abstention
decision, the answer, and whether each quote is actually in the passage it cites.

This is the manual error-inspection tool: before a metric says a configuration is
better, look at what it retrieves for a handful of questions.

Run:
    python scripts/ask.py "Wie oft muss die Interne Revision prüfen?"
    python scripts/ask.py "..." --no-generate          # retrieval only, no API call
    python scripts/ask.py "..." --method bm25 --k 10
"""
import argparse
import textwrap

from regrag import config, storage
from regrag.evaluate import RelevanceJudge
from regrag.generate import Generator
from regrag.pipeline import answer_question, load_retriever
from regrag.retrieve import should_abstain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--method", choices=("dense", "bm25", "hybrid"), default=config.RETRIEVER)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--threshold", type=float, default=config.RELEVANCE_THRESHOLD)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    manifest = storage.read_json(config.MANIFEST_PATH) if config.MANIFEST_PATH.exists() else {}
    embedder = None
    if manifest.get("has_dense_index"):
        from regrag.embed import Embedder

        embedder = Embedder(manifest["embedding_model"])
    retriever = load_retriever(args.method, embedder)
    hits = retriever.search(args.question, args.k)

    for h in hits:
        c = h.chunk
        dense = f"  cos {h.dense_score:.3f}" if h.dense_score is not None else ""
        print(f"\n[{h.rank}] score {h.score:.3f}{dense}   {c.chunk_id}   {c.section_id} {c.section_title}"
              f"   p. {c.page_start}-{c.page_end}")
        print(textwrap.indent(textwrap.shorten(c.text, 400, placeholder=" ..."), "    "))

    abstain = should_abstain(hits, args.threshold)
    print(f"\nretrieval threshold {args.threshold}: {'ABSTAIN' if abstain else 'pass'}")
    if args.no_generate or abstain:
        return

    answer, _ = answer_question(args.question, retriever, Generator(), args.k, args.threshold, hits=hits)
    if answer.error:
        print(f"\nerror: {answer.error}")
        return
    print(f"\nanswerable: {answer.answerable}   (served by {answer.model}, "
          f"{answer.input_tokens} in / {answer.output_tokens} out tokens)")
    print("\n" + textwrap.fill(answer.answer, 100))
    print(f"\ncitations: {answer.citations}")
    judge = RelevanceJudge(retriever.chunks)
    for q in answer.quotes:
        p = q["passage"]
        cov = judge.coverage(hits[p - 1].chunk, q["text"]) if 1 <= p <= len(hits) else 0.0
        mark = "ok " if cov >= config.QUOTE_MIN_COVERAGE else "NOT FOUND"
        print(f"  [{p}] {mark} ({cov:.0%})  {textwrap.shorten(q['text'], 140)}")


if __name__ == "__main__":
    main()
