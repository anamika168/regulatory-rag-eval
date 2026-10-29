"""Dense (FAISS), lexical (BM25) and hybrid retrieval behind one interface.

The corpus is a few hundred to a few thousand chunks, so the FAISS index is exact
(IndexFlatIP). An approximate index would add a second source of misses to an
evaluation whose purpose is to explain misses.

Whatever the ranking method, every hit carries the dense cosine score of its chunk when
an embedder is available, because abstention is decided on that score: BM25 scores are
unbounded and depend on query length, so a fixed threshold on them means nothing.
"""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from regrag import config
from regrag.chunk import Chunk
from regrag.text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    rank: int                       # 1-based
    score: float                    # score of the ranking method
    dense_score: float | None       # cosine similarity, None without an embedder


# --- Dense ----------------------------------------------------------------

class DenseIndex:
    def __init__(self, vectors: np.ndarray):
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.vectors = vectors

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        k = min(k, len(self.vectors))
        return self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Cosine score of one query against every chunk."""
        return self.vectors @ query

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: Path) -> "DenseIndex":
        import faiss

        index = faiss.read_index(str(path))
        return cls(index.reconstruct_n(0, index.ntotal))


# --- Lexical --------------------------------------------------------------

class BM25Index:
    """Okapi BM25. Small enough to write out, which keeps its behaviour inspectable."""

    def __init__(self, texts: list[str], k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=np.float64)
        self.avg_length = float(self.lengths.mean()) if len(self.docs) else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avg_length or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out


# --- Combined -------------------------------------------------------------

def reciprocal_rank_fusion(rankings: list[list[int]], k: int = config.RRF_K) -> list[tuple[int, float]]:
    """Fuse ranked lists of chunk indices; returns (index, fused score), best first."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, chunks: list[Chunk], method: str = config.RETRIEVER, embedder=None,
                 dense: DenseIndex | None = None):
        if method not in ("dense", "bm25", "hybrid"):
            raise ValueError(f"unknown retrieval method {method!r}")
        if method in ("dense", "hybrid") and embedder is None:
            raise ValueError(f"method {method!r} needs an embedder")
        self.chunks = chunks
        self.method = method
        self.embedder = embedder
        self.dense = dense
        if embedder is not None and dense is None:
            self.dense = DenseIndex(embedder.encode_passages([c.embed_text for c in chunks]))
        if self.dense is not None and len(self.dense.vectors) != len(chunks):
            raise ValueError("index and chunk list differ in length -- rebuild the index")
        self.bm25 = BM25Index([c.embed_text for c in chunks]) if method in ("bm25", "hybrid") else None

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        return self.search_many([query], k)[0]

    def search_many(self, queries: list[str], k: int = config.TOP_K) -> list[list[Hit]]:
        """Batch search; embedding all queries at once is much faster than one by one."""
        q_vectors = self.embedder.encode_queries(queries) if self.embedder is not None else None
        results = []
        for i, query in enumerate(queries):
            dense_scores = self.dense.scores(q_vectors[i]) if q_vectors is not None else None
            if self.method == "dense":
                order = np.argsort(-dense_scores, kind="stable")[:k]
                ranked = [(int(j), float(dense_scores[j])) for j in order]
            elif self.method == "bm25":
                lexical = self.bm25.scores(query)
                order = np.argsort(-lexical, kind="stable")[:k]
                ranked = [(int(j), float(lexical[j])) for j in order]
            else:
                pool = config.HYBRID_POOL
                lexical = self.bm25.scores(query)
                ranked = reciprocal_rank_fusion([
                    [int(j) for j in np.argsort(-dense_scores, kind="stable")[:pool]],
                    [int(j) for j in np.argsort(-lexical, kind="stable")[:pool]],
                ])[:k]
            results.append([
                Hit(chunk=self.chunks[j], rank=r, score=s,
                    dense_score=float(dense_scores[j]) if dense_scores is not None else None)
                for r, (j, s) in enumerate(ranked, start=1)
            ])
        return results


def should_abstain(hits: list[Hit], threshold: float = config.RELEVANCE_THRESHOLD) -> bool:
    """Retrieval-level abstention: nothing retrieved is similar enough to the question.

    Without dense scores (pure BM25 run) there is no calibrated signal, and the decision
    is left to the generator.
    """
    scores = [h.dense_score for h in hits if h.dense_score is not None]
    if not scores:
        return not hits
    return max(scores) < threshold
