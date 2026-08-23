"""Central configuration.

Every tunable that an experiment might vary lives here, so that an ablation is a
change to one value rather than a hunt through the codebase. The defaults below are
starting points, not conclusions -- the point of the evaluation harness is to find
out which of them are wrong.
"""

from pathlib import Path

# --- Paths ----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_DIR = ROOT / "data" / "eval"

INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

# --- Chunking -------------------------------------------------------------

# Target chunk size in characters, not tokens: German legal text has long compound
# nouns, so a token-based target drifts in physical size across documents.
CHUNK_SIZE = 1200

# Overlap exists so that a sentence spanning a boundary is not lost to both chunks.
# Too much overlap inflates the index and makes retrieval return near-duplicates.
CHUNK_OVERLAP = 200

# Markers that START a section. These are the valid chunk boundaries.
# Verified against BaFin Rundschreiben 06/2024 and 06/2026 by scripts/probe_extraction.py;
# evidence recorded in docs/findings_session02.tex.
# BTD and BTK were assumed initially and occur zero times in both documents -- removed.
STRUCTURE_MARKERS = ("AT", "BTO", "BTR", "BT")

# Markers that REFERENCE a section from inside running text ("gemäß Tz. 5").
# Tz. occurs 33 and 41 times but begins a line only twice, and never, respectively.
# These must NOT be used as chunk boundaries: splitting here cuts sentences at citations.
# That § is likewise a citation (to other statutes) is still a hypothesis, not measured.
CROSS_REFERENCE_MARKERS = ("Tz.", "§")
# --- Embeddings -----------------------------------------------------------

# Multilingual model, chosen so that German queries and German passages land in the
# same space without a translation step. Runs locally on CPU; no API cost.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 768

# --- Retrieval ------------------------------------------------------------

TOP_K = 5

# Cosine similarity below which a passage is treated as "not really relevant".
# This threshold is what makes abstention possible: if nothing clears it, the honest
# answer is that the corpus does not cover the question. Calibrating it is a core
# experiment, not a constant to be guessed once.
RELEVANCE_THRESHOLD = 0.35

# --- Generation -----------------------------------------------------------

# Claude Opus 5 is the default. For bulk evaluation runs where cost matters more than
# per-answer quality, claude-haiku-4-5 is roughly 5x cheaper on input and output;
# comparing the two across the metric suite is itself a worthwhile experiment.
GENERATION_MODEL = "claude-opus-5"
MAX_TOKENS = 2048

# --- Evaluation -----------------------------------------------------------

# k values at which recall is reported. Reporting a single k hides whether a miss is a
# ranking problem (answer at position 8) or a retrieval problem (answer nowhere).
RECALL_AT_K = (1, 3, 5, 10)
