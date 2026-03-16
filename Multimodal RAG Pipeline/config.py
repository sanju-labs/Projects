"""
config.py — Single source of truth for all settings.
Change anything here; the rest of the codebase adapts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
QDRANT_PATH = PROJECT_ROOT / "qdrant_data"
IMAGE_CACHE = PROJECT_ROOT / ".image_cache"

# Create dirs if missing
DATA_DIR.mkdir(exist_ok=True)
QDRANT_PATH.mkdir(exist_ok=True)
IMAGE_CACHE.mkdir(exist_ok=True)

# ── API Keys ─────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Models ───────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # $0.02 / 1M tokens — best cost/quality
EMBEDDING_DIMS = 1536                        # Dimensions for text-embedding-3-small
LLM_MODEL = "gpt-4.1-mini"                  # Fast, cheap, vision-capable
LLM_TEMPERATURE = 0.1                        # Low temp = factual answers

# ── Chunking ─────────────────────────────────────────
CHUNK_SIZE = 1000           # Characters per chunk (sweet spot for retrieval)
CHUNK_OVERLAP = 200         # Overlap to preserve context across boundaries

# ── Retrieval ────────────────────────────────────────
QDRANT_COLLECTION = "multimodal_docs"
TOP_K = 5                   # Number of chunks to retrieve per query
HYBRID_SEARCH = True        # Combine semantic + keyword (BM25) search
SEMANTIC_WEIGHT = 0.7       # Weight for semantic search (0-1)
KEYWORD_WEIGHT = 0.3        # Weight for keyword/BM25 search (0-1)

# ── Chat Memory ──────────────────────────────────────
MAX_HISTORY_TURNS = 5       # How many Q&A pairs to keep for follow-ups
ENABLE_QUERY_REWRITE = True # Rewrite follow-up questions with context

# ── Semantic Cache ───────────────────────────────────
CACHE_COLLECTION = "semantic_cache"
CACHE_SIMILARITY_THRESHOLD = 0.92   # Cosine similarity ≥ this = cache hit
CACHE_ENABLED = True                # Toggle caching on/off

# ── Vision ───────────────────────────────────────────
MIN_IMAGE_SIZE = 100        # width or height in px (skip smaller)
MAX_IMAGE_DESCRIBE_TOKENS = 300  # Max tokens for image description

# ── Table Extraction ─────────────────────────────────
EXTRACT_TABLES = True       # Extract tables as structured markdown

# ── Auto Ingest ──────────────────────────────────────
AUTO_INGEST_ON_STARTUP = True   # Auto-ingest PDFs in data/ when app starts
