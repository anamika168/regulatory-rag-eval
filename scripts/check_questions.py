#!/usr/bin/env python3
"""
Validate data/eval/questions.yaml against the extracted corpus, before any metric is run.

For every gold_span of every answerable question, reports:
  FOUND    the span occurs (after normalisation) in the named document, with page and section
  PARTIAL  only part of it occurs in one piece -- usually a wording difference; the closest
           original text is printed so that the span can be corrected by copying it
  MISSING  nothing close was found
and flags spans that also occur in another document (near-duplicates across versions:
if the question is version-specific, source_doc must be set) and gold_section labels
that disagree with where the span was actually found.

A gold span that is not in the extracted text cannot be retrieved by any system, so
every MISSING span would be counted as a retrieval failure that is really a labelling
or extraction failure. Exit code 1 if any span is not FOUND.

Run:
    python scripts/check_questions.py
"""
import sys
from difflib import SequenceMatcher

from regrag.chunk import build_chunks
from regrag.evaluate import RelevanceJudge, load_questions
from regrag.pipeline import load_or_ingest_pages
from regrag.text import key_with_offsets, match_key


def main() -> int:
    try:
        questions = load_questions()
    except ValueError as exc:
        print(exc)
        return 1
    pages = load_or_ingest_pages()
    docs = sorted({p.doc for p in pages})
    page_keys = [(p, *key_with_offsets(p.text)) for p in pages]
    # Section lookup works on structure-aware chunks regardless of the configured strategy.
    judge = RelevanceJudge(build_chunks(pages, "structure"))
    sections = {c.chunk_id: c.section_id for c in judge.chunks}

    print(f"{len(questions)} questions, {sum(q.answerable for q in questions)} answerable; "
          f"corpus: {', '.join(docs)}\n")
    for d in {q.source_doc for q in questions if q.source_doc} - set(docs):
        print(f"source_doc not in corpus: {d}")

    problems = 0
    for q in questions:
        for span in q.gold_spans:
            span_key = match_key(span)
            in_docs = sorted({p.doc for p, key, _ in page_keys if span_key in key})
            # Spans across a page break are not in any single page; search joined text too.
            if not in_docs:
                in_docs = [d for d in docs if span_key in "".join(k for p, k, _ in page_keys if p.doc == d)]
            target_docs = [q.source_doc] if q.source_doc else docs
            found = [d for d in in_docs if d in target_docs]
            if found:
                hit_pages = [p.page for p, key, _ in page_keys if p.doc in found and span_key in key]
                ids = judge.relevant_chunks(span, q.source_doc)
                found_sections = sorted({sections[i] for i in ids})
                line = f"FOUND    {q.id:<12} {', '.join(found)} p. {hit_pages or '(across page break)'}"
                line += f" section {', '.join(found_sections) or '?'}"
                if q.gold_section and q.gold_section not in found_sections:
                    line += f"   <-- gold_section says {q.gold_section}"
                    problems += 1
                if q.source_doc and set(in_docs) - {q.source_doc}:
                    line += f"   (also in {', '.join(sorted(set(in_docs) - {q.source_doc}))})"
                print(line)
                continue

            problems += 1
            best = None
            for p, key, offsets in page_keys:
                if p.doc not in target_docs or not key:
                    continue
                m = SequenceMatcher(None, key, span_key, autojunk=False).find_longest_match(
                    0, len(key), 0, len(span_key))
                if best is None or m.size > best[0]:
                    best = (m.size, p, m.a, offsets)
            coverage = best[0] / len(span_key) if best else 0.0
            status = "PARTIAL " if coverage >= 0.3 else "MISSING "
            print(f"{status} {q.id:<12} best coverage {coverage:.0%}"
                  + (f" on {best[1].doc} p. {best[1].page}" if best else ""))
            print(f"           span:    {span}")
            if best and best[0]:
                size, page, a, offsets = best
                start = max(0, offsets[a] - 120)
                end = min(len(page.text), offsets[min(a + size, len(offsets)) - 1] + 200)
                print(f"           nearest: ...{page.text[start:end]}...".replace("\n", " "))
    print(f"\n{problems} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
