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
RESULTS_DIR = ROOT / "results"

QUESTIONS_PATH = EVAL_DIR / "questions.yaml"
SOURCES_PATH = ROOT / "data" / "sources.yaml"

PAGES_PATH = PROCESSED_DIR / "pages.jsonl"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
# Records how the index was built. A query embedded with a different model than the one
# that built the index returns garbage without any error, so this is checked on load.
MANIFEST_PATH = PROCESSED_DIR / "index_manifest.json"

# --- Ingest ---------------------------------------------------------------

# "plain"   -- pdfplumber's default reading order, as used by scripts/probe_extraction.py.
# "columns" -- split the page at the gap between requirement text and the Erläuterungen
#              column and re-attach each explanation to the Textziffer it stands next to.
# "auto"    -- "columns" on pages where a column gap is detected, "plain" elsewhere.
# Plain extraction interleaves the two columns line by line, which produces sentences
# that exist in neither column. Whether "auto" detects the gap on every page is checked
# by scripts/probe_extraction.py, not assumed.
EXTRACTION_LAYOUT = "auto"

# The column gap is searched only in this horizontal band (fractions of page width), so
# that the page margins and a narrow Textziffer margin column are never taken for it.
COLUMN_SEARCH_BAND = (0.30, 0.85)

# Minimum width in points of a vertical whitespace band to count as a column gap.
COLUMN_GAP_MIN_WIDTH = 8.0

# Each side of the gap must hold at least this share of the page's words (and the right
# side at least three lines); otherwise the "gap" is just a short line or an indented
# block. Kept low because some pages carry only one or two short Erläuterungen, and
# reading such a page in plain mode would splice those into the requirement text.
COLUMN_MIN_SIDE_SHARE = 0.05

# Prefix put in front of text from the right-hand column, so that both the embedding
# and the generator can tell an explanation from the binding requirement.
EXPLANATION_PREFIX = "Erläuterung:"

# Page furniture: a line among the first/last FURNITURE_EDGE_LINES lines of a page that
# recurs (digits ignored) on at least FURNITURE_MIN_PAGE_SHARE of pages is a running
# header or footer and is removed.
FURNITURE_EDGE_LINES = 3
FURNITURE_MIN_PAGE_SHARE = 0.5

# A page on which at least TOC_MIN_ENTRIES lines look like "AT 4.1 Title .... 12" is a
# table-of-contents page and is excluded, because its entries match the heading pattern.
TOC_MIN_ENTRIES = 5

# --- Chunking -------------------------------------------------------------

# "structure" -- sections split on STRUCTURE_MARKERS, packed by whole Textziffern.
# "fixed"     -- character windows of CHUNK_SIZE with CHUNK_OVERLAP, ignoring structure.
# "fixed" exists as the baseline the structure-aware strategy has to beat.
CHUNK_STRATEGY = "structure"

# Target chunk size in characters, not tokens: German legal text has long compound
# nouns, so a token-based target drifts in physical size across documents.
CHUNK_SIZE = 1200

# Overlap exists so that a sentence spanning a boundary is not lost to both chunks.
# Too much overlap inflates the index and makes retrieval return near-duplicates.
# Structure-aware chunks only use it when a single Textziffer exceeds CHUNK_SIZE.
CHUNK_OVERLAP = 200

# A structure-aware chunk shorter than this share of CHUNK_SIZE is merged into the
# previous chunk of the same section instead of standing alone.
MIN_CHUNK_SHARE = 0.25

# Prepend "AT 4.4.2 Compliance-Funktion" (with parent headings) to the text that gets
# embedded. A Textziffer often never names its own topic; the heading does.
PREPEND_HEADING = True

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

# Models compared in the ablation. E5 models were trained with "query: " / "passage: "
# prefixes and degrade without them; the paraphrase models were trained without.
ABLATION_EMBEDDING_MODELS = (
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-base",
)
EMBEDDING_PREFIXES = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}
EMBEDDING_BATCH_SIZE = 32

# --- Retrieval ------------------------------------------------------------

# "dense"  -- FAISS inner product over normalised embeddings (= cosine similarity).
# "bm25"   -- lexical baseline; no model, so it shows what embeddings actually add.
# "hybrid" -- reciprocal-rank fusion of both lists.
RETRIEVER = "dense"

TOP_K = 5

# Cosine similarity below which a passage is treated as "not really relevant".
# This threshold is what makes abstention possible: if nothing clears it, the honest
# answer is that the corpus does not cover the question. Calibrating it is a core
# experiment, not a constant to be guessed once.
RELEVANCE_THRESHOLD = 0.35

# Thresholds reported by the abstention sweep in scripts/evaluate_retrieval.py.
THRESHOLD_SWEEP = tuple(round(0.20 + 0.05 * i, 2) for i in range(11))  # 0.20 ... 0.70

BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal-rank fusion constant; 60 is the value from the original RRF paper and
# damps the influence of the very first ranks. HYBRID_POOL is how deep each list is read.
RRF_K = 60
HYBRID_POOL = 50

# --- Generation -----------------------------------------------------------

# Claude Opus 5 is the default. For bulk evaluation runs where cost matters more than
# per-answer quality, claude-haiku-4-5 is roughly 5x cheaper on input and output;
# comparing the two across the metric suite is itself a worthwhile experiment.
GENERATION_MODEL = "claude-opus-5"

# Thinking tokens count towards this limit, so it is set well above the length of the
# answer itself; a truncated response is recorded as an error, not parsed.
MAX_TOKENS = 16000

# None leaves effort at the model's default. Set "low" / "medium" / "high" to make it an
# experiment variable. claude-haiku-4-5 rejects the parameter, so None is the safe default.
GENERATION_EFFORT = None

# Server-side fallback re-runs a declined request on another model. Off by default: an
# evaluation run that silently mixes two models measures neither. Declines are recorded
# as errors instead, and the serving model is stored with every answer either way.
ALLOW_MODEL_FALLBACK = False

# Model for the optional groundedness judge. Deliberately not the generation model, so
# that the judge is not grading its own phrasing.
JUDGE_MODEL = "claude-sonnet-5-5"

# --- Evaluation -----------------------------------------------------------

# k values at which recall is reported. Reporting a single k hides whether a miss is a
# ranking problem (answer at position 8) or a retrieval problem (answer nowhere).
RECALL_AT_K = (1, 3, 5, 10)

# A chunk counts as containing a gold span if it holds at least this share of the span
# as one contiguous piece (after normalisation). Below 1.0 so that a span cut by a chunk
# boundary still counts for the chunk that holds most of it; this keeps relevance
# defined by the text, independent of which chunker produced the chunk.
SPAN_MIN_COVERAGE = 0.6

# A quote in a generated answer counts as grounded if at least this share of it appears
# contiguously in the passage it cites. Close to 1.0: a quote is meant to be verbatim.
QUOTE_MIN_COVERAGE = 0.9
