# regulatory-rag-eval

Retrieval-Augmented Generation over German financial-regulatory texts, with an emphasis on
**measuring retrieval quality and knowing when to abstain** rather than on producing fluent answers.

> **Status: work in progress, started 23 August 2026.**
> This README describes the intended scope and records what is actually working so far.
> The progress table below is kept honest — nothing is listed as done until it runs end to end.

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
| Ingest | PDF text extraction, layout cleanup | How much structure survives extraction? |
| Chunk | Structure-aware splitting on legal markers | Does splitting on § / Absatz beat fixed-size windows? |
| Embed | Multilingual sentence-transformers | How well do multilingual embeddings handle German legal register? |
| Retrieve | FAISS dense retrieval, top-k | Where does nearest-neighbour diverge from relevant? |
| Generate | Claude, with retrieved context injected | Does the answer stay inside the evidence? |
| Evaluate | recall@k, MRR, groundedness, abstention rate | The point of the repository |

## Progress

- [x] Repository scaffold, dependencies, configuration
- [ ] Corpus acquisition and text extraction
- [ ] Structure-aware chunking
- [ ] Embeddings and FAISS index
- [ ] Retrieval and manual error inspection
- [ ] Generation with context injection
- [ ] Labelled evaluation set
- [ ] Retrieval metrics (recall@k, MRR)
- [ ] Groundedness and abstention metrics
- [ ] Ablations: chunk size, k, embedding model
- [ ] Failure analysis write-up

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # only needed from the generation stage onwards
```

## Layout

```
src/regrag/      pipeline modules (ingest, chunk, embed, retrieve, generate, evaluate)
data/raw/        source PDFs (not version-controlled)
data/processed/  extracted text and indices (not version-controlled)
data/eval/       labelled question set (version-controlled - this is the valuable part)
scripts/         corpus fetching and pipeline entry points
tests/           unit tests
```

## Limitations

Recorded as they are found, not at the end. Nothing here is production code.

## Licence

MIT
