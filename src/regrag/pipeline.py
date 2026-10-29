"""Glue between the stages: build or load an index, answer one question end to end."""

from datetime import datetime, timezone

from regrag import config, storage
from regrag.chunk import Chunk, build_chunks
from regrag.generate import Answer, Generator
from regrag.ingest import Page, corpus_paths, ingest_corpus
from regrag.retrieve import DenseIndex, Hit, Retriever, should_abstain


def load_or_ingest_pages(refresh: bool = False) -> list[Page]:
    """Pages from data/processed/pages.jsonl, re-extracting the PDFs if needed."""
    if config.PAGES_PATH.exists() and not refresh:
        return storage.load_pages(config.PAGES_PATH)
    paths = corpus_paths()
    if not paths:
        raise SystemExit(f"No PDFs found in {config.RAW_DIR}")
    pages, _ = ingest_corpus(paths)
    storage.save_pages(config.PAGES_PATH, pages)
    return pages


def save_index(chunks: list[Chunk], dense: DenseIndex | None, settings: dict) -> None:
    storage.save_chunks(config.CHUNKS_PATH, chunks)
    if dense is not None:
        dense.save(config.INDEX_PATH)
    elif config.INDEX_PATH.exists():
        config.INDEX_PATH.unlink()      # a stale index next to new chunks would be misread
    manifest = dict(settings, n_chunks=len(chunks), has_dense_index=dense is not None,
                    built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    storage.write_json(config.MANIFEST_PATH, manifest)


def load_retriever(method: str = config.RETRIEVER, embedder=None) -> Retriever:
    """Retriever over the saved index; refuses an embedder that did not build it."""
    if not config.MANIFEST_PATH.exists():
        raise SystemExit("No index found. Run scripts/build_index.py first.")
    manifest = storage.read_json(config.MANIFEST_PATH)
    chunks = storage.load_chunks(config.CHUNKS_PATH)
    dense = None
    if method in ("dense", "hybrid"):
        if not manifest.get("has_dense_index"):
            raise SystemExit("The saved index has no embeddings. Rebuild without --no-dense.")
        if embedder is not None and embedder.name != manifest["embedding_model"]:
            raise SystemExit(f"Index was built with {manifest['embedding_model']}, "
                             f"but queries would be embedded with {embedder.name}.")
        dense = DenseIndex.load(config.INDEX_PATH)
    elif embedder is not None and manifest.get("has_dense_index") and embedder.name == manifest["embedding_model"]:
        dense = DenseIndex.load(config.INDEX_PATH)     # BM25 ranking, dense scores for abstention
    return Retriever(chunks, method=method, embedder=embedder if dense is not None else None, dense=dense)


def answer_question(question: str, retriever: Retriever, generator: Generator | None,
                    k: int = config.TOP_K, threshold: float = config.RELEVANCE_THRESHOLD,
                    hits: list[Hit] | None = None) -> tuple[Answer, list[Hit]]:
    """Retrieve, then either abstain on retrieval scores or let the generator answer.

    Abstaining before the API call is not only cheaper: it keeps the abstention decision
    measurable on its own, separately from the model's.
    """
    if hits is None:
        hits = retriever.search(question, k)
    if should_abstain(hits, threshold):
        return Answer(question=question, answerable=None,
                      answer="Die vorliegenden Dokumente enthalten dazu keine ausreichend passende Stelle.",
                      passage_ids=[h.chunk.chunk_id for h in hits], abstained_by="retrieval"), hits
    if generator is None:
        raise ValueError("a generator is needed to answer questions that pass the threshold")
    return generator.answer(question, hits), hits


def build(pages: list[Page], strategy: str = config.CHUNK_STRATEGY, size: int = config.CHUNK_SIZE,
          overlap: int = config.CHUNK_OVERLAP, embedder=None) -> tuple[list[Chunk], DenseIndex | None]:
    chunks = build_chunks(pages, strategy, size, overlap)
    dense = DenseIndex(embedder.encode_passages([c.embed_text for c in chunks])) if embedder else None
    return chunks, dense
