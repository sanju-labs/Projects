"""
rag_chain.py — Retrieval-Augmented Generation chain.

Query → Cache check → (if miss) Hybrid search → Context → LLM → Cache store.
Also generates "Behind the Scenes" (BTS) metadata for each query.
"""

import re
import time
import base64
from pathlib import Path
from collections import Counter

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

import config
import semantic_cache

# ── Clients ──────────────────────────────────────────
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
qdrant_client = QdrantClient(path=str(config.QDRANT_PATH))
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY,
)

# ── System Prompt ────────────────────────────────────
SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions based on retrieved document context.

RULES:
1. Answer ONLY from the provided context. If the context doesn't contain the answer, say so clearly.
2. When referencing information, mention the source document and page number.
3. If the context includes image descriptions, use that visual information naturally.
4. If the context includes tables, reference specific data from the table.
5. Be concise but thorough.
6. If multiple sources conflict, mention the discrepancy.
"""


# ─────────────────────────────────────────────────────
# Query rewriting for follow-ups
# ─────────────────────────────────────────────────────
def rewrite_query(query: str, chat_history: list[dict]) -> str:
    if not config.ENABLE_QUERY_REWRITE or not chat_history:
        return query

    signals = ["it", "this", "that", "they", "them", "those",
               "the same", "more about", "what about", "how about",
               "and ", "also", "previous", "above", "earlier"]
    q_lower = query.lower()
    is_followup = any(s in q_lower for s in signals)

    if not is_followup and len(query.split()) >= 8:
        return query

    history_str = ""
    for msg in chat_history[-config.MAX_HISTORY_TURNS * 2:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        history_str += f"{role}: {content}\n"

    response = openai_client.chat.completions.create(
        model=config.LLM_MODEL, temperature=0, max_tokens=150,
        messages=[
            {"role": "system", "content": (
                "Rewrite the user's follow-up question as a standalone search query. "
                "Output ONLY the rewritten query — no explanation."
            )},
            {"role": "user", "content": f"CONVERSATION:\n{history_str}\nFOLLOW-UP: {query}\nREWRITTEN:"},
        ],
    )
    return response.choices[0].message.content.strip() or query


# ─────────────────────────────────────────────────────
# Sparse vector for keyword search
# ─────────────────────────────────────────────────────
def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_sparse_vector(text):
    tokens = _tokenize(text)
    if not tokens:
        return SparseVector(indices=[0], values=[1.0])
    freq = Counter(tokens)
    return SparseVector(
        indices=[abs(hash(t)) % (2**31) for t in freq],
        values=[float(c) for c in freq.values()],
    )


# ─────────────────────────────────────────────────────
# RRF Fusion
# ─────────────────────────────────────────────────────
def rrf_fuse(dense_results, sparse_results, k=60):
    dw, sw = config.SEMANTIC_WEIGHT, config.KEYWORD_WEIGHT
    scores = {}
    for rank, hit in enumerate(dense_results):
        scores.setdefault(hit.id, {"score": 0, "payload": hit.payload})
        scores[hit.id]["score"] += dw * (1.0 / (k + rank + 1))
    for rank, hit in enumerate(sparse_results):
        scores.setdefault(hit.id, {"score": 0, "payload": hit.payload})
        scores[hit.id]["score"] += sw * (1.0 / (k + rank + 1))

    fused = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [{
        "text": d["payload"].get("text", ""),
        "metadata": {
            "source": d["payload"].get("source", "unknown"),
            "page": d["payload"].get("page", "?"),
            "content_type": d["payload"].get("content_type", "text"),
            "image_path": d["payload"].get("image_path", ""),
        },
        "score": round(d["score"], 4),
    } for _, d in fused]


# ─────────────────────────────────────────────────────
# Retrieve with timing metadata
# ─────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = None) -> tuple[list[dict], dict]:
    """
    Returns (chunks, bts_meta) where bts_meta has timing and search details.
    """
    k = top_k or config.TOP_K
    bts = {}

    # Embed query
    t0 = time.time()
    query_vector = embeddings.embed_query(query)
    bts["embed_time_ms"] = round((time.time() - t0) * 1000, 1)
    bts["query_vector_dims"] = len(query_vector)
    bts["query_vector_sample"] = [round(v, 4) for v in query_vector[:5]]

    # Count total chunks in DB
    try:
        info = qdrant_client.get_collection(config.QDRANT_COLLECTION)
        bts["total_chunks_in_db"] = info.points_count
    except Exception:
        bts["total_chunks_in_db"] = 0

    t1 = time.time()

    if config.HYBRID_SEARCH:
        dense_results = qdrant_client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=("dense", query_vector), limit=k * 2,
        )
        sparse_vec = compute_sparse_vector(query)
        sparse_results = qdrant_client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=("sparse", sparse_vec), limit=k * 2,
        )
        bts["search_time_ms"] = round((time.time() - t1) * 1000, 1)
        bts["dense_candidates"] = len(dense_results)
        bts["sparse_candidates"] = len(sparse_results)
        bts["sparse_terms"] = len(sparse_vec.indices)
        bts["search_type"] = "hybrid"

        fused = rrf_fuse(dense_results, sparse_results)
        chunks = fused[:k]
    else:
        results = qdrant_client.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=("dense", query_vector), limit=k,
        )
        bts["search_time_ms"] = round((time.time() - t1) * 1000, 1)
        bts["dense_candidates"] = len(results)
        bts["sparse_candidates"] = 0
        bts["search_type"] = "semantic_only"

        chunks = [{
            "text": h.payload.get("text", ""),
            "metadata": {
                "source": h.payload.get("source", "unknown"),
                "page": h.payload.get("page", "?"),
                "content_type": h.payload.get("content_type", "text"),
                "image_path": h.payload.get("image_path", ""),
            },
            "score": round(h.score, 4),
        } for h in results]

    bts["top_k"] = k
    bts["chunks_returned"] = len(chunks)
    bts["chunk_types"] = {}
    for c in chunks:
        ct = c["metadata"]["content_type"]
        bts["chunk_types"][ct] = bts["chunk_types"].get(ct, 0) + 1
    bts["top_scores"] = [c["score"] for c in chunks]

    return chunks, bts


# ─────────────────────────────────────────────────────
# Build context and messages
# ─────────────────────────────────────────────────────
def build_context(chunks):
    ICONS = {"text": "📝 Text", "table": "📊 Table", "image": "🖼️ Image"}
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        tag = ICONS.get(m["content_type"], "📝 Text")
        parts.append(f"--- Chunk {i} ({tag}) [Source: {m['source']}, Page {m['page']}] ---\n{c['text']}")
    return "\n\n".join(parts)


def build_messages(query, chunks):
    context = build_context(chunks)
    blocks = []
    for c in chunks:
        if c["metadata"]["content_type"] == "image":
            p = c["metadata"].get("image_path", "")
            if p and Path(p).exists():
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                blocks.append({"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}", "detail": "low"
                }})
    blocks.append({"type": "text", "text": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": blocks}]


def _extract_sources(chunks):
    seen = set()
    sources = []
    for c in chunks:
        key = (c["metadata"]["source"], c["metadata"]["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "source": c["metadata"]["source"], "page": c["metadata"]["page"],
                "type": c["metadata"]["content_type"], "score": c["score"],
            })
    return sources


# ─────────────────────────────────────────────────────
# Build BTS (Behind the Scenes) explanation text
# ─────────────────────────────────────────────────────
def build_bts_explanation(query: str, search_query: str, bts: dict, sources: list, cache_hit: bool, cache_similarity: float = 0) -> str:
    """
    Generate a human-readable BTS explanation using ONLY actual pipeline data.
    No AI is used here — just formatting the real numbers.
    """
    tokens = re.findall(r"[a-z0-9]+", search_query.lower())
    char_count = len(search_query)

    lines = []
    lines.append("### 🔍 Step 1 — Your query")
    lines.append(f'You typed: **"{query}"**')
    if search_query != query:
        lines.append(f'The system detected a follow-up and rewrote it to: **"{search_query}"** — this standalone version retrieves better results.')
    lines.append(f"Your query has **{char_count} characters** and **{len(tokens)} tokens** (words/numbers after lowercasing and cleaning).")
    lines.append("")

    if cache_hit:
        lines.append("### ⚡ Step 2 — Semantic cache HIT")
        lines.append(f"Before doing any search, the system checked if a similar question was asked before. It found a match with **{cache_similarity:.1%} similarity** (threshold: {config.CACHE_SIMILARITY_THRESHOLD:.0%}).")
        lines.append("Since the match is above the threshold, the cached answer was returned **instantly** — no embedding, no database search, no LLM call. This saved time and cost.")
        lines.append("")
        lines.append("### 🏁 Result")
        lines.append("Answer served from cache. Zero API calls for this query.")
        return "\n".join(lines)

    lines.append("### 🧬 Step 2 — Embedding your query")
    lines.append(f"Your query was converted into a **{bts['query_vector_dims']}-dimensional vector** (a list of {bts['query_vector_dims']} decimal numbers) using OpenAI's `{config.EMBEDDING_MODEL}` model. This took **{bts['embed_time_ms']}ms**.")
    lines.append(f"First 5 values of your query vector: `{bts['query_vector_sample']}`")
    lines.append("This vector captures the *meaning* of your question — similar questions produce similar vectors, even with different wording.")
    lines.append("")

    if bts["search_type"] == "hybrid":
        lines.append("### 🔎 Step 3 — Hybrid search (semantic + keyword)")
        lines.append(f"The database has **{bts['total_chunks_in_db']} chunks** stored from your documents.")
        lines.append("")
        lines.append("**Semantic search (70% weight):** Your query vector was compared against every chunk's vector using cosine similarity. Think of it as asking *\"which chunks mean something similar to my question?\"* — this found **{0} candidate chunks**.".format(bts['dense_candidates']))
        lines.append("")
        lines.append(f"**Keyword search (30% weight):** Your query was also broken into **{bts['sparse_terms']} unique terms** and matched against chunks containing the same words. This is like traditional search — exact word matching. It found **{bts['sparse_candidates']} candidate chunks**.")
        lines.append("")
        lines.append(f"**Reciprocal Rank Fusion (RRF):** Both ranked lists were merged. A chunk that ranks high in *both* searches gets a higher combined score than one that ranks high in only one. This fusion took **{bts['search_time_ms']}ms** total.")
    else:
        lines.append("### 🔎 Step 3 — Semantic search")
        lines.append(f"The database has **{bts['total_chunks_in_db']} chunks**. Your query vector was compared against all of them using cosine similarity. Found **{bts['dense_candidates']} candidates** in **{bts['search_time_ms']}ms**.")

    lines.append("")
    lines.append(f"### 📦 Step 4 — Top-K selection (K={bts['top_k']})")
    lines.append(f"From all the candidates, the **top {bts['chunks_returned']} chunks** were selected:")

    type_breakdown = []
    for ct, count in bts["chunk_types"].items():
        icon = {"text": "📝", "table": "📊", "image": "🖼️"}.get(ct, "📝")
        type_breakdown.append(f"{icon} {count} {ct} chunk{'s' if count != 1 else ''}")
    lines.append("  " + " • ".join(type_breakdown))
    lines.append("")
    lines.append("Relevance scores (higher = more relevant):")
    for i, score in enumerate(bts["top_scores"], 1):
        bar_len = int(score * 200) if score < 0.1 else int(min(score * 40, 30))
        bar = "█" * max(bar_len, 1)
        lines.append(f"  Chunk {i}: `{score:.4f}` {bar}")

    lines.append("")
    lines.append("These chunks form the *context window* — the information the LLM will read to answer your question.")
    lines.append("")

    lines.append(f"### 🧠 Step 5 — LLM generation (`{config.LLM_MODEL}`)")
    lines.append(f"The {bts['chunks_returned']} retrieved chunks were assembled into a context block, along with any referenced images (sent as actual image data for visual grounding).")
    lines.append(f"The LLM reads this context + your question and generates an answer. It's instructed to **only use the provided context** — no guessing, no hallucination. Temperature is set to **{config.LLM_TEMPERATURE}** (low = factual, deterministic).")
    lines.append("")

    lines.append("### 💾 Step 6 — Cache storage")
    lines.append(f"The query + answer pair was stored in the semantic cache. If you (or anyone) asks a similar question later (≥{config.CACHE_SIMILARITY_THRESHOLD:.0%} similarity), the cached answer will be returned instantly — no search or LLM needed.")
    lines.append("")

    lines.append("### 📚 Sources used")
    for s in sources:
        icon = {"text": "📄", "table": "📊", "image": "🖼️"}.get(s["type"], "📄")
        lines.append(f"  {icon} **{s['source']}** — Page {s['page']} (score: {s['score']})")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────
# ask() — Main entry point with cache + BTS
# ─────────────────────────────────────────────────────
def ask(query: str, top_k: int = None, chat_history: list = None, generate_bts: bool = False) -> dict:
    """
    Full RAG pipeline with semantic cache.

    Returns: {
        answer, sources, rewritten_query,
        cache_hit, cache_similarity,
        bts_text (if generate_bts=True)
    }
    """
    # Step 0: Rewrite if follow-up
    search_query = rewrite_query(query, chat_history or [])

    # Step 1: Check semantic cache
    cached = semantic_cache.lookup(search_query)
    if cached:
        result = {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "rewritten_query": search_query if search_query != query else None,
            "cache_hit": True,
            "cache_similarity": cached["similarity"],
        }
        if generate_bts:
            result["bts_text"] = build_bts_explanation(
                query, search_query, {}, cached["sources"],
                cache_hit=True, cache_similarity=cached["similarity"]
            )
        return result

    # Step 2: Retrieve (with BTS metadata)
    chunks, bts_meta = retrieve(search_query, top_k)

    if not chunks:
        return {
            "answer": "No documents found. Please add PDFs to the `data/` folder and restart.",
            "sources": [], "rewritten_query": search_query,
            "cache_hit": False, "cache_similarity": 0,
        }

    # Step 3: Generate answer
    messages = build_messages(query, chunks)
    response = openai_client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=1024,
        messages=messages,
    )
    answer = response.choices[0].message.content
    sources = _extract_sources(chunks)

    # Step 4: Store in cache
    semantic_cache.store(search_query, answer, sources)

    result = {
        "answer": answer,
        "sources": sources,
        "rewritten_query": search_query if search_query != query else None,
        "cache_hit": False,
        "cache_similarity": 0,
    }

    if generate_bts:
        result["bts_text"] = build_bts_explanation(
            query, search_query, bts_meta, sources,
            cache_hit=False,
        )

    return result


# ─────────────────────────────────────────────────────
# ask_stream() — Streaming version with cache + BTS
# ─────────────────────────────────────────────────────
def ask_stream(query: str, top_k: int = None, chat_history: list = None, generate_bts: bool = False):
    """
    Streaming version. Yields events:
      {"type": "cache_hit", "data": {...}}
      {"type": "rewritten", "data": "..."}
      {"type": "sources", "data": [...]}
      {"type": "token", "data": "..."}
      {"type": "bts", "data": "..."}
      {"type": "done"}
    """
    search_query = rewrite_query(query, chat_history or [])
    if search_query != query:
        yield {"type": "rewritten", "data": search_query}

    # Check cache
    cached = semantic_cache.lookup(search_query)
    if cached:
        yield {"type": "cache_hit", "data": {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "similarity": cached["similarity"],
        }}
        if generate_bts:
            bts_text = build_bts_explanation(
                query, search_query, {}, cached["sources"],
                cache_hit=True, cache_similarity=cached["similarity"],
            )
            yield {"type": "bts", "data": bts_text}
        yield {"type": "done"}
        return

    # Retrieve
    chunks, bts_meta = retrieve(search_query, top_k)
    if not chunks:
        yield {"type": "token", "data": "No documents found. Please add PDFs to the `data/` folder and restart."}
        yield {"type": "done"}
        return

    sources = _extract_sources(chunks)
    yield {"type": "sources", "data": sources}

    # Stream answer
    messages = build_messages(query, chunks)
    stream = openai_client.chat.completions.create(
        model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE,
        max_tokens=1024, messages=messages, stream=True,
    )

    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            yield {"type": "token", "data": delta.content}

    # Cache the result
    semantic_cache.store(search_query, full_text, sources)

    # BTS
    if generate_bts:
        bts_text = build_bts_explanation(
            query, search_query, bts_meta, sources, cache_hit=False,
        )
        yield {"type": "bts", "data": bts_text}

    yield {"type": "done"}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    print(f"\n🔎 Query: {q}\n")
    result = ask(q, generate_bts=True)
    if result["cache_hit"]:
        print(f"⚡ Cache HIT (similarity: {result['cache_similarity']})")
    print(f"💬 Answer:\n{result['answer']}\n")
    if result.get("bts_text"):
        print(f"\n--- BTS ---\n{result['bts_text']}")
