#!/usr/bin/env python3
"""
End-to-end evaluation: retrieve, abstain or generate, then score abstention and grounding.

Calls the Anthropic API once per question that passes the retrieval threshold (twice
with --judge). Use --dry-run first: it prints one prompt and the number of calls a run
would make, without making any.

Writes to results/:
  generation.jsonl   one record per question (written as the run goes, so an interrupted
                     run keeps what it has)
  generation.md      summary and the cases worth reading: answered-but-unanswerable,
                     ungrounded quotes, false abstentions

Run:
    python scripts/evaluate_generation.py --dry-run
    python scripts/evaluate_generation.py --limit 5
    python scripts/evaluate_generation.py --judge
"""
import argparse
import json

from regrag import config, report, storage
from regrag.evaluate import RelevanceJudge, evaluate_answer, load_questions, summarize_generation
from regrag.generate import Answer, Generator, Judge, build_prompt
from regrag.pipeline import answer_question, load_retriever
from regrag.retrieve import should_abstain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("dense", "bm25", "hybrid"), default=config.RETRIEVER)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--threshold", type=float, default=config.RELEVANCE_THRESHOLD)
    parser.add_argument("--model", default=config.GENERATION_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--only", nargs="+", metavar="ID", help="only these question ids")
    parser.add_argument("--judge", action="store_true", help="add the LLM groundedness judge")
    parser.add_argument("--dry-run", action="store_true", help="show one prompt and the call count, call nothing")
    parser.add_argument("--out", default="generation", help="output file stem in results/")
    args = parser.parse_args()

    from regrag.embed import Embedder

    manifest = storage.read_json(config.MANIFEST_PATH) if config.MANIFEST_PATH.exists() else {}
    embedder = Embedder(manifest["embedding_model"]) if manifest.get("has_dense_index") else None
    retriever = load_retriever(args.method, embedder)

    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q.id in set(args.only)]
    if args.limit:
        questions = questions[:args.limit]
    all_hits = retriever.search_many([q.question for q in questions], args.k)
    to_generate = [q for q, hits in zip(questions, all_hits) if not should_abstain(hits, args.threshold)]

    print(f"{len(questions)} questions; {len(questions) - len(to_generate)} stopped by the retrieval threshold "
          f"{args.threshold}; {len(to_generate)} go to {args.model}"
          + (f" (+{len(to_generate)} judge calls to {config.JUDGE_MODEL})" if args.judge else ""))
    if args.dry_run:
        if to_generate:
            q = to_generate[0]
            print(f"\n--- prompt for {q.id} ---\n{build_prompt(q.question, all_hits[questions.index(q)])}")
        return

    generator = Generator(model=args.model)
    judge_model = Judge() if args.judge else None
    relevance = RelevanceJudge(retriever.chunks)
    rows = []
    out = config.RESULTS_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for q, hits in zip(questions, all_hits):
            try:
                answer, hits = answer_question(q.question, retriever, generator, args.k, args.threshold, hits=hits)
            except Exception as exc:     # one failed request must not end a long run
                answer = Answer(question=q.question, answerable=None, answer="",
                                passage_ids=[h.chunk.chunk_id for h in hits], error=f"{type(exc).__name__}: {exc}")
            row = evaluate_answer(q, answer, hits, relevance)
            if judge_model and not answer.abstained and not answer.error:
                try:
                    row["judge"] = judge_model.judge(q.question, answer, hits)
                except Exception as exc:
                    row["judge"] = {"verdict": None, "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            status = row["error"] or (f"abstained ({row['abstained_by']})" if row["abstained"]
                                      else f"answered, {row.get('n_grounded_quotes', 0)}/{row.get('n_quotes', 0)} quotes grounded")
            print(f"  {q.id:<12} {status}")

    summary = summarize_generation(rows)
    settings = {"method": args.method, "k": args.k, "threshold": args.threshold, "generation_model": args.model,
                "effort": config.GENERATION_EFFORT, "judge_model": config.JUDGE_MODEL if args.judge else None,
                "embedding_model": manifest.get("embedding_model"), "chunk_strategy": manifest.get("chunk_strategy"),
                "chunk_size": manifest.get("chunk_size")}
    served = sorted({r["model"] for r in rows if r.get("model")})
    if served and served != [args.model]:
        settings["served_by"] = ", ".join(served)

    md = ["# Generation evaluation\n", "\n## Settings\n\n", report.key_values(settings),
          "\n## Summary\n\n", report.key_values(summary)]
    fabricated = [r for r in rows if not r["answerable_gold"] and not r["abstained"] and not r["error"]]
    md += [f"\n## Answered although the corpus has no answer ({len(fabricated)})\n\n",
           report.table([{"id": r["id"], "question": r["question"], "answer": r["answer"]} for r in fabricated])]
    ungrounded = [r for r in rows if r.get("n_quotes") and r["n_grounded_quotes"] < r["n_quotes"]]
    md += [f"\n## Answers with quotes not found in the cited passage ({len(ungrounded)})\n\n",
           report.table([{"id": r["id"], "quote coverage": ", ".join(f"{c:.2f}" for c in r["quote_coverage"]),
                          "quotes": " / ".join(q["text"] for q in r["quotes"])} for r in ungrounded])]
    false_abst = [r for r in rows if r["answerable_gold"] and r["abstained"]]
    md += [f"\n## False abstentions ({len(false_abst)})\n\n",
           report.table([{"id": r["id"], "by": r["abstained_by"], "gold retrieved": r["retrieved_gold"],
                          "question": r["question"]} for r in false_abst])]
    errors = [r for r in rows if r["error"]]
    if errors:
        md += [f"\n## Errors ({len(errors)})\n\n", report.table([{"id": r["id"], "error": r["error"]} for r in errors])]
    out.with_suffix(".md").write_text("".join(md), encoding="utf-8")

    print("\n" + report.key_values(summary))
    print(f"written: {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
