# regulatory-rag-eval

Retrieval-Augmented Generation over German financial-regulatory texts, with an emphasis on
**measuring retrieval quality and knowing when to abstain** rather than on producing fluent answers.

> **Status: work in progress, started 23 August 2026.**
> This README describes the intended scope and records what is actually working so far.
> The progress table below is kept honest — nothing is listed as run until it runs end to end on the corpus.

## Why this project

Most RAG demonstrations stop at "the system produced plausible-looking text". That is the easy half.
The hard half, and the half that decides whether such a system can be deployed in a regulated setting,
is answering three questions with numbers:

1. **Did retrieval actually find the passage that contains the answer?** Fluent generation on top of
   the wrong passage is worse than no answer, because it is confidently wrong.
2. **Is the generated answer grounded in the retrieved text**, or has the model filled gaps from its
   own parametric knowledge?
3. **Does the system abstain when the corpus does not contain the answer?** In a regulatory context a
   confident fabrication is the most expensive possible failure mode.

This repository treats those as the primary objects of study. Generation quality is downstream.

## Domain

German financial-regulatory publications (BaFin circulars, MaRisk, Solvency II material). Chosen
because the language is precise, the documents have strong internal structure (§, Absatz, Randziffer)
that a chunking strategy can exploit, and because answers are checkable against a specific passage
rather than a matter of taste.

## Approach

| Stage | Method | Design question being tested |
|---|---|---|
| Ingest | pdfplumber, column-aware reading, header/footer and TOC removal | How much structure survives extraction? |
| Chunk | Sections split on AT / BT / BTO / BTR headings, packed by whole Textziffern; fixed-size baseline | Does splitting on structure beat fixed-size windows? |
| Embed | Multilingual sentence-transformers, normalised vectors | How well do multilingual embeddings handle German legal register? |
| Retrieve | FAISS exact search; BM25 baseline; reciprocal-rank fusion | Where does nearest-neighbour diverge from relevant? |
| Generate | Claude, retrieved passages as numbered context, JSON-schema output with citations and verbatim quotes | Does the answer stay inside the evidence? |
| Evaluate | recall@k, MRR, abstention rates, quote grounding, optional LLM judge | The point of the repository |

## Progress

| Stage | Implemented and unit-tested | Run on the corpus |
|---|---|---|
| Repository scaffold, dependencies, configuration | yes | — |
| Corpus acquisition and text extraction | yes | probe only |
| Structure-aware chunking | yes | not yet |
| Embeddings and FAISS index | yes | not yet |
| Retrieval and manual error inspection | yes (`scripts/ask.py`) | not yet |
| Generation with context injection | yes | not yet |
| Labelled evaluation set | 30 answerable + 8 unanswerable drafted | spans not yet checked against the extracted text |
| Retrieval metrics (recall@k, MRR) | yes | not yet |
| Groundedness and abstention metrics | yes | not yet |
| Ablations: chunk size, k, embedding model | yes | not yet |
| Failure analysis write-up | tooling in place (miss tables, per-question records) | after the corpus run |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                     # makes `regrag` importable from scripts/
export ANTHROPIC_API_KEY=...        # only needed from the generation stage onwards
```

The PDFs go into `data/raw/` (not version-controlled). `python scripts/fetch_corpus.py --record` writes their
SHA-256 into `data/sources.yaml`; add the source URLs there by hand.

## Running it

```bash
python scripts/probe_extraction.py      # recon: marker counts, text layer, detected column split per page
python scripts/build_index.py           # ingest -> chunk -> embed -> data/processed/
python scripts/check_questions.py       # every gold span must be FOUND before any metric is trusted
python scripts/evaluate_retrieval.py    # recall@k, MRR, abstention sweep, misses -> results/
python scripts/evaluate_retrieval.py --method bm25
python scripts/run_ablations.py         # strategy x size x embedding model x method, retrieval only
python scripts/ask.py "Wie oft muss die Interne Revision prüfen?" --no-generate
python scripts/evaluate_generation.py --dry-run     # prints one prompt and the number of API calls
python scripts/evaluate_generation.py               # end to end, then --judge for the LLM judge
pytest
```

Everything up to and including the ablations runs locally without API calls.

## How evaluation works

**Relevance is defined by text, not by chunk ids.** Each question in `data/eval/questions.yaml` carries a
short verbatim `gold_span`. A chunk is relevant if it contains at least `SPAN_MIN_COVERAGE` (60 %) of that span
as one contiguous piece, after normalising case, whitespace, hyphenation and quote characters. Because the labels
never refer to a chunk, the same question set scores the structure-aware chunker, the fixed-size baseline and
every chunk size, without relabelling and without favouring the configuration the labels were made with.

**Retrieval.** recall@k is the share of a question's gold spans found in the top k (1, 3, 5, 10); MRR uses the
rank of the first relevant chunk. Both are reported over all answerable questions and over those whose gold span
exists in the index at all — the difference between the two is what extraction and chunking lost before
retrieval had a chance. `section_hit@k` (was the right section retrieved, even if not the right chunk) separates
ranking problems inside a section from retrieval problems.

**Abstention, at two levels.**
1. *Retrieval:* if no retrieved passage reaches `RELEVANCE_THRESHOLD` cosine similarity, the system abstains
   without calling the model. `evaluate_retrieval.py` reports how well the top score separates answerable from
   unanswerable questions (AUC) and a threshold sweep. The sweep is fitted on the questions it is reported on,
   so it is read as a curve, not as a tuned number.
2. *Generation:* the model returns `answerable: false` when the passages do not contain the answer. The
   unanswerable questions include near misses (e.g. a required staff number for the Compliance-Funktion) whose
   passages score high, so a threshold cannot stop them; only the generator can.

**Groundedness.** The model must return verbatim quotes per cited passage. A quote counts as grounded if at
least `QUOTE_MIN_COVERAGE` (90 %) of it occurs contiguously in the passage it names. This is a string check,
not a second model's opinion. An answer is *fully grounded* if all its quotes are grounded and every cited passage
is quoted. `--judge` adds a claim-level check by a different model as a secondary signal.

## Layout

```
src/regrag/
  config.py       every tunable; an ablation is a change to one value
  text.py         normalisation for display (clean_text) and for matching (match_key), BM25 tokens
  ingest.py       PDF -> pages: column split, explanation re-attachment, furniture and TOC removal
  chunk.py        sections and Textziffern; structure-aware and fixed-size chunkers
  embed.py        sentence-transformers wrapper, truncation report
  retrieve.py     FAISS dense index, BM25, reciprocal-rank fusion, threshold abstention
  generate.py     prompt, JSON-schema output, answer parsing; optional judge
  pipeline.py     build / load the index, answer one question end to end
  evaluate.py     question set, span-based relevance, all metrics
  report.py, storage.py   markdown tables, JSONL
data/raw/         source PDFs (not version-controlled)
data/processed/   extracted pages, chunks, index and its manifest (not version-controlled)
data/eval/        labelled question set (version-controlled - this is the valuable part)
data/sources.yaml corpus provenance: file, URL, SHA-256
scripts/          probe, build, check, evaluate, ablate, ask, fetch
results/          evaluation output as JSONL and markdown
tests/            unit tests on synthetic text; no model download, no API calls
```

## Limitations

Recorded as they are found, not at the end. Nothing here is production code.

- **The question set is small.** With 30 answerable questions, one question is more than three points of
  recall. Differences of one or two questions between ablation rows are not evidence of anything.
- **Abstention threshold and question set are not independent.** There is no held-out split; see above.
- **The column reader is a heuristic.** It assumes requirement text left, Erläuterungen right, aligned by
  Textziffer. Pages where `probe_extraction.py` shows no split are read in plain mode.
- **Textziffer detection relies on sequence.** A line starting with the next expected number ("3 Monate ...")
  is taken as a Textziffer. The tests document this case rather than hide it.
- **Embedding input limits.** Chunks longer than the model's `max_seq_length` are embedded only in part;
  `build_index.py` prints how many.
- **Quote grounding checks verbatim-ness, not entailment.** A real sentence quoted in support of a claim it
  does not support passes the string check; that is what the judge is for, and the judge is itself unvalidated.

## Licence

MIT
