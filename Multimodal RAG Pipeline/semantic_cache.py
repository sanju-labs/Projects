"""
semantic_cache.py — Semantic caching layer.

If a new query is semantically similar (cosine ≥ threshold) to a cached query,
return the cached answer instantly — no embedding, no retrieval, no LLM call.

How it works:
  1. Embed the query
  2. Search the cache collection (separate Qdrant collection)
  3. If top hit ≥ threshold → cache HIT → return stored answer
  4. If no match → cache MISS → proceed with full RAG pipeline
  5. After getting the answer, store (query_embedding, answer, sources) in cache
"""

import time
import json

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)

import config

# ── Clients ──────────────────────────────────────────
qdrant_client = QdrantClient(path=str(config.QDRANT_PATH))
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY,
)


def _ensure_cache_collection():
    """Create cache collection if it doesn't exist."""
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if config.CACHE_COLLECTION not in collections:
        qdrant_client.create_collection(
            collection_name=config.CACHE_COLLECTION,
            vectors_config=VectorParams(
                size=config.EMBEDDING_DIMS,
                distance=Distance.COSINE,
            ),
        )


def lookup(query: str) -> dict | None:
    """
    Check if a semantically similar query exists in cache.

    Returns:
        dict with {answer, sources, cached_query, similarity} if HIT
        None if MISS
    """
    if not config.CACHE_ENABLED:
        return None

    _ensure_cache_collection()

    # Embed the query
    query_vector = embeddings.embed_query(query)

    # Search cache
    results = qdrant_client.search(
        collection_name=config.CACHE_COLLECTION,
        query_vector=query_vector,
        limit=1,
    )

    if not results:
        return None

    top = results[0]
    similarity = top.score

    if similarity >= config.CACHE_SIMILARITY_THRESHOLD:
        return {
            "answer": top.payload.get("answer", ""),
            "sources": json.loads(top.payload.get("sources_json", "[]")),
            "cached_query": top.payload.get("query", ""),
            "similarity": round(similarity, 4),
        }

    return None


def store(query: str, answer: str, sources: list[dict]):
    """
    Store a query-answer pair in the semantic cache.
    """
    if not config.CACHE_ENABLED:
        return

    _ensure_cache_collection()

    query_vector = embeddings.embed_query(query)

    # Get next ID
    try:
        info = qdrant_client.get_collection(config.CACHE_COLLECTION)
        next_id = info.points_count
    except Exception:
        next_id = 0

    qdrant_client.upsert(
        collection_name=config.CACHE_COLLECTION,
        points=[
            PointStruct(
                id=next_id,
                vector=query_vector,
                payload={
                    "query": query,
                    "answer": answer,
                    "sources_json": json.dumps(sources),
                    "timestamp": time.time(),
                },
            )
        ],
    )


def get_cache_stats() -> dict:
    """Return cache size info."""
    try:
        info = qdrant_client.get_collection(config.CACHE_COLLECTION)
        return {"cached_queries": info.points_count}
    except Exception:
        return {"cached_queries": 0}


def clear_cache():
    """Delete the cache collection."""
    try:
        qdrant_client.delete_collection(config.CACHE_COLLECTION)
    except Exception:
        pass
