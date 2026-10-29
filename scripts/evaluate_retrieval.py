#!/usr/bin/env python3
"""
Retrieval evaluation over data/eval/questions.yaml against the saved index. No API calls.

Writes to results/:
  retrieval_<method>.jsonl   one record per question: retrieved chunks, relevance, ranks
  retrieval_<method>.md      summary, abstention sweep, and every miss listed by question

The summary reports recall@k and MRR twice: over all answerable questions, and over
those whose gold span exists in the index at all. The difference is what extraction and
chunking lost before retrieval had a chance.

Run:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --method bm25
    python scripts/evaluate_retrieval.py --method hybrid
"""
import argparse

from regrag import config, report, storage
from regrag.evaluate import (RelevanceJudge, evaluate_retrieval_question, load_questions, separability,
                             summarize_retrieval, threshold_sweep)
from regrag.pipeline import load_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("dense", "bm25", "hybrid"), default=config.RETRIEVER)
    parser.add_argument("--no-embedder", action="store_true",
                        help="BM25 without loading the embedding model (no dense scores, no abstention sweep)")
    args = parser.parse_args()

    manifest = storage.read_json(config.MANIFEST_PATH) if config.MANIFEST_PATH.exists() else {}
    embedder = None
    if not args.no_embedder and manifest.get("has_dense_index"):
        from regrag.embed import Embedder

        embedder = Embedder(manifest["embedding_model"])
    retriever = load_retriever(args.method, embedder)

    questions = load_questions()
    depth = max(config.RECALL_AT_K)
    all_hits = retriever.search_many([q.question for q in questions], depth)
    judge = RelevanceJudge(retriever.chunks)
    rows = [evaluate_retrieval_question(q, hits, judge) for q, hits in zip(questions, all_hits)]
    summary = summarize_retrieval(rows)
    sweep = threshold_sweep(rows)
    sep = separability(rows)

    out = config.RESULTS_DIR / f"retrieval_{args.method}"
    storage.write_jsonl(out.with_suffix(".jsonl"), rows)

    ks = config.RECALL_AT_K
    md = [f"# Retrieval evaluation: {args.method}\n",
          "\n## Index\n\n", report.key_values({k: manifest.get(k) for k in
                                            ("chunk_strategy", "chunk_size", "chunk_overlap", "extraction_layout",
                                             "embedding_model", "n_chunks", "built_at")}),
          "\n## Summary\n\n", report.key_values(summary)]
    if sep.get("auc") is not None:
        md += ["\n## Does the top score separate answerable from unanswerable?\n\n",
               report.key_values({"AUC": sep["auc"],
                                  "answerable top score (min / median / max)":
                                      " / ".join(str(v) for v in sep["answerable_score"].values()),
                                  "unanswerable top score (min / median / max)":
                                      " / ".join(str(v) for v in sep["unanswerable_score"].values())}),
               f"\n## Abstention threshold sweep (k = {config.TOP_K})\n\n",
               "Fitted and reported on the same questions -- read as a curve, not as a tuned value.\n\n",
               report.table(sweep)]

    answerable = [r for r in rows if r["answerable"]]
    misses = [r for r in answerable if r[f"recall@{config.TOP_K}"] == 0]
    md += [f"\n## Misses at k = {config.TOP_K} ({len(misses)} of {len(answerable)})\n\n",
           "`gold in index = no` means the miss happened before retrieval (extraction, chunking or a wrong "
           "gold span -- run scripts/check_questions.py).\n\n",
           report.table([{
               "id": r["id"], "gold section": r["gold_section"],
               "gold in index": r["spans_in_index"] == r["n_gold_spans"],
               "first relevant rank": r["first_relevant_rank"],
               "retrieved sections (top 3)": ", ".join(h["section_id"] for h in r["retrieved"][:3]),
               "top score": r["top_dense_score"],
               "question": r["question"],
           } for r in misses])]
    unanswerable = sorted((r for r in rows if not r["answerable"]),
                          key=lambda r: -(r["top_dense_score"] or 0))
    md += ["\n## Unanswerable questions, most confidently matched first\n\n",
           "The top of this list cannot be stopped by a threshold without also stopping answerable questions; "
           "for these, abstention has to come from the generator.\n\n",
           report.table([{"id": r["id"], "top score": r["top_dense_score"],
                          "retrieved sections (top 3)": ", ".join(h["section_id"] for h in r["retrieved"][:3]),
                          "question": r["question"]} for r in unanswerable])]
    md += ["\n## Per question\n\n",
           report.table([{"id": r["id"], **{f"r@{k}": r.get(f"recall@{k}") for k in ks},
                          "rr": r.get("reciprocal_rank"), "top score": r["top_dense_score"]} for r in rows])]
    out.with_suffix(".md").write_text("".join(md), encoding="utf-8")

    print(report.key_values(summary))
    if sep.get("auc") is not None:
        print(f"separability AUC: {sep['auc']}")
    print(f"written: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
