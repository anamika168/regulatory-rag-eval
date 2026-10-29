#!/usr/bin/env python3
"""
Retrieval ablations: chunking strategy x chunk size x embedding model x retrieval method.

Retrieval only, so no API cost; every configuration is scored on the same questions with
the same span-based relevance, which is what makes the rows comparable. Pages are read
from data/processed/pages.jsonl (run build_index.py once first); nothing is written to
the saved index.

Overlap scales with chunk size (size // 6, i.e. 200 at 1200) so that a smaller chunk
does not end up mostly overlap.

"structure-noheading" is the structure-aware chunker without the heading prepended to
the embedded text. Without it, structure vs fixed would compare two changes at once:
where the boundaries are, and whether the chunk carries its section title.

Writes results/ablations.jsonl and results/ablations.md.

Run:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --sizes 600 1200 --methods dense bm25
    python scripts/run_ablations.py --models none        # BM25 only, no model download
"""
import argparse
import time

from regrag import config, report, storage
from regrag.chunk import build_chunks
from regrag.evaluate import (RelevanceJudge, evaluate_retrieval_question, load_questions, separability,
                             summarize_retrieval)
from regrag.pipeline import load_or_ingest_pages
from regrag.retrieve import DenseIndex, Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=["structure", "structure-noheading", "fixed"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[600, 1200, 2000])
    parser.add_argument("--models", nargs="+", default=list(config.ABLATION_EMBEDDING_MODELS),
                        help="embedding models; 'none' skips dense and hybrid")
    parser.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    args = parser.parse_args()

    pages = load_or_ingest_pages()
    questions = load_questions()
    queries = [q.question for q in questions]
    depth = max(config.RECALL_AT_K)
    models = [m for m in args.models if m != "none"]
    dense_methods = [m for m in args.methods if m in ("dense", "hybrid")]

    embedders = {}
    if models and dense_methods:
        from regrag.embed import Embedder

        embedders = {m: Embedder(m) for m in models}

    results = []
    for strategy in args.strategies:
        for size in args.sizes:
            overlap = size // 6
            chunks = build_chunks(pages, strategy.removesuffix("-noheading"), size, overlap)
            if strategy.endswith("-noheading"):
                for c in chunks:
                    c.heading_path = []
            judge = RelevanceJudge(chunks)
            configs = []
            if "bm25" in args.methods:
                configs.append(("bm25", None))
            configs += [(method, m) for m in embedders for method in dense_methods]

            dense_cache: dict[str, tuple[DenseIndex, dict]] = {}
            for method, model in configs:
                started = time.time()
                embedder = embedders.get(model)
                dense, trunc = None, {}
                if embedder is not None:
                    if model not in dense_cache:
                        texts = [c.embed_text for c in chunks]
                        dense_cache[model] = (DenseIndex(embedder.encode_passages(texts)),
                                              embedder.truncation_report(texts))
                    dense, trunc = dense_cache[model]
                retriever = Retriever(chunks, method=method, embedder=embedder, dense=dense)
                hits = retriever.search_many(queries, depth)
                rows = [evaluate_retrieval_question(q, h, judge) for q, h in zip(questions, hits)]
                summary = summarize_retrieval(rows)
                result = {
                    "strategy": strategy, "size": size, "overlap": overlap,
                    "model": model.split("/")[-1] if model else "-", "method": method,
                    "n_chunks": len(chunks), "gold_in_index": summary["n_gold_in_index"],
                    **{f"recall@{k}": summary[f"all_recall@{k}"] for k in config.RECALL_AT_K},
                    "mrr": summary["all_mrr"],
                    "auc": separability(rows)["auc"] if embedder is not None else None,
                    "truncated": trunc.get("share_truncated"),
                }
                results.append(result)
                print(f"{strategy:<9} {size:>5} {result['model']:<40} {method:<6} "
                      f"r@{config.TOP_K}={result[f'recall@{config.TOP_K}']} mrr={result['mrr']} "
                      f"({time.time() - started:.0f}s)")

    storage.write_jsonl(config.RESULTS_DIR / "ablations.jsonl", results)
    ranked = sorted(results, key=lambda r: (-(r[f"recall@{config.TOP_K}"] or 0), -(r["mrr"] or 0)))
    md = ["# Retrieval ablations\n\n",
          f"{len(questions)} questions ({sum(q.answerable for q in questions)} answerable). Recall and MRR over "
          "all answerable questions; `gold_in_index` = questions whose gold spans survive chunking; "
          "`auc` = separability of answerable vs unanswerable by top dense score; `truncated` = share of chunks "
          "longer than the embedding model's input limit.\n\n",
          "With this few questions, one question is several percentage points of recall: differences of one or "
          "two questions between rows are not evidence of anything.\n\n",
          report.table(ranked)]
    (config.RESULTS_DIR / "ablations.md").write_text("".join(md), encoding="utf-8")
    print(f"written: {config.RESULTS_DIR / 'ablations.md'}")


if __name__ == "__main__":
    main()
