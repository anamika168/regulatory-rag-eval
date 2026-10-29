#!/usr/bin/env python3
"""
Ingest the PDFs in data/raw, chunk them, embed the chunks and write the index.

Writes to data/processed/:
  pages.jsonl           cleaned page texts -- read some of these before trusting anything downstream
  chunks.jsonl          chunks with section, page range and Textziffern
  faiss.index           dense index (skipped with --no-dense)
  index_manifest.json   the settings the index was built with

Prints, per document, what ingest removed (TOC pages, running headers) and on how many
pages the column layout was detected; then chunk statistics; then how many chunks are
longer than the embedding model's input limit and are therefore embedded only in part.

Run:
    python scripts/build_index.py
    python scripts/build_index.py --strategy fixed --chunk-size 800 --overlap 150
    python scripts/build_index.py --no-dense          # BM25 only, no model download
"""
import argparse
from collections import Counter

import numpy as np

from regrag import config, storage
from regrag.chunk import build_chunks
from regrag.ingest import corpus_paths, ingest_corpus
from regrag.pipeline import save_index
from regrag.retrieve import DenseIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=("structure", "fixed"), default=config.CHUNK_STRATEGY)
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--layout", choices=("auto", "columns", "plain"), default=config.EXTRACTION_LAYOUT)
    parser.add_argument("--model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--no-dense", action="store_true", help="skip embeddings (BM25-only runs)")
    args = parser.parse_args()

    paths = corpus_paths()
    if not paths:
        raise SystemExit(f"No PDFs found in {config.RAW_DIR}")

    pages, reports = ingest_corpus(paths, args.layout)
    storage.save_pages(config.PAGES_PATH, pages)
    for r in reports:
        print(f"\n===== {r.doc} =====")
        print(f"pages: {r.n_pages}, kept: {r.n_pages - len(r.empty_pages) - len(r.toc_pages)}")
        print(f"  empty (no text layer): {r.empty_pages or '-'}")
        print(f"  table of contents:     {r.toc_pages or '-'}")
        print(f"  read as two columns:   {len(r.column_pages)} pages")
        print(f"  running headers/footers removed: {r.furniture_lines or '-'}")

    chunks = build_chunks(pages, args.strategy, args.chunk_size, args.overlap)
    lengths = np.array([len(c.text) for c in chunks])
    print(f"\n===== chunks ({args.strategy}, size {args.chunk_size}, overlap {args.overlap}) =====")
    for doc, n in Counter(c.doc for c in chunks).items():
        print(f"  {doc}: {n} chunks, {len({c.section_id for c in chunks if c.doc == doc})} sections")
    print(f"  length in chars: min {lengths.min()}, median {int(np.median(lengths))}, "
          f"p95 {int(np.percentile(lengths, 95))}, max {lengths.max()}")
    if args.strategy == "structure":
        print(f"  chunks without a Textziffer: {sum(not c.textziffern for c in chunks)} "
              f"(preamble, or a section whose numbering was not recognised)")

    dense = None
    settings = {"chunk_strategy": args.strategy, "chunk_size": args.chunk_size, "chunk_overlap": args.overlap,
                "extraction_layout": args.layout, "prepend_heading": config.PREPEND_HEADING,
                "embedding_model": None, "documents": [p.name for p in paths]}
    if not args.no_dense:
        from regrag.embed import Embedder

        embedder = Embedder(args.model)
        texts = [c.embed_text for c in chunks]
        trunc = embedder.truncation_report(texts)
        print(f"\n===== embeddings ({embedder.name}) =====")
        print(f"  input limit {trunc['max_seq_length']} tokens; median chunk {trunc['median_tokens']}, "
              f"max {trunc['max_tokens']}")
        print(f"  truncated: {trunc['n_truncated']} of {trunc['n_passages']} chunks "
              f"({trunc['share_truncated']:.1%}) -- their tail is not part of the embedding")
        dense = DenseIndex(embedder.encode_passages(texts))
        settings.update(embedding_model=embedder.name, embedding_dim=dense.dim, truncation=trunc)

    save_index(chunks, dense, settings)
    print(f"\nwritten: {config.PAGES_PATH.name}, {config.CHUNKS_PATH.name}"
          f"{', ' + config.INDEX_PATH.name if dense is not None else ''}, {config.MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()
