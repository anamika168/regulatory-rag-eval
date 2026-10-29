"""Sentence-transformer embeddings.

Vectors are L2-normalised, so that inner product in FAISS equals cosine similarity and
RELEVANCE_THRESHOLD has the same meaning for every model.

The truncation report matters more than it looks: sentence-transformers silently cuts
every input at the model's max_seq_length. paraphrase-multilingual-mpnet-base-v2 stops
at 128 word pieces, and a 1200-character German chunk is usually longer than that, so
the end of such a chunk never reaches the embedding at all. build_index prints how many
chunks are affected for the model in use.
"""

import numpy as np

from regrag import config


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL, device: str | None = None,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE):
        # Imported here: torch takes seconds to load, and BM25-only runs and the unit
        # tests do not need it.
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = config.EMBEDDING_PREFIXES.get(model_name, ("", ""))

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_seq_length(self) -> int:
        return self.model.max_seq_length

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return vectors.astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])

    def truncation_report(self, texts: list[str]) -> dict:
        """How many passages exceed the model's input length and are cut."""
        tokenizer = self.model.tokenizer
        lengths = [len(tokenizer.encode(self.passage_prefix + t, add_special_tokens=True)) for t in texts]
        limit = self.max_seq_length
        over = [n for n in lengths if n > limit]
        return {
            "model": self.name,
            "max_seq_length": limit,
            "n_passages": len(lengths),
            "n_truncated": len(over),
            "share_truncated": round(len(over) / len(lengths), 3) if lengths else 0.0,
            "median_tokens": int(np.median(lengths)) if lengths else 0,
            "max_tokens": max(lengths) if lengths else 0,
        }
