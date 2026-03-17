# 🤖 Hi, I'm Bing! — Multimodal RAG

**Drop your PDFs. Ask anything. See how the magic works.**

A production-grade RAG system that reads text, understands tables, sees images, caches smartly, and explains itself. Built with LangChain, Qdrant, and GPT-4.1 mini.

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## 60-Second Setup

```bash
# Clone it
git clone https://github.com/YOUR_USERNAME/multimodal-rag.git
cd multimodal-rag

# Run the setup script (creates venv, installs everything)
chmod +x setup.sh && ./setup.sh

# Add your OpenAI key
nano .env   # paste your key

# Drop your PDFs
cp ~/Documents/my-report.pdf data/

# Launch
source .venv/bin/activate
streamlit run app.py
```

That's it. The app auto-ingests your PDFs on first launch. No upload buttons, no manual steps.

---

## How It Works

### The simple version

1. **You drop PDFs into `data/`** — the app reads them automatically on startup
2. **You ask a question** — the system finds the most relevant pieces and generates an answer
3. **Ask again** — if it's similar to a previous question, you get an instant cached answer

### The detailed version (what the BTS toggle shows you)

When you toggle **"Curious about BTS? Click to explore!"** in the sidebar, every answer comes with a real-time breakdown of what happened inside the system — using actual numbers from your query, not AI-generated explanations.

Here's what happens under the hood:

```
Your question
     │
     ▼
Query Rewriter (expands follow-ups like "what about that?")
     │
     ▼
Semantic Cache Check ──→ HIT? Return instantly, skip everything below
     │ MISS
     ▼
Embed query → 1,536-dimensional vector
     │
     ├──→ Semantic search (cosine similarity, 70% weight)
     └──→ Keyword search (BM25 term matching, 30% weight)
            │
            ▼
     RRF Fusion (merges both ranked lists)
            │
            ▼
     Top-5 chunks + any referenced images
            │
            ▼
     GPT-4.1 mini reads context + generates answer
            │
            ▼
     Answer cached for future similar questions
```

---

## Features

### Semantic Cache

If you ask "What was Q3 revenue?" and later ask "How much revenue in Q3?" — the system recognizes these are the same question (≥92% cosine similarity) and returns the cached answer instantly. Zero API calls, zero cost, zero latency.

### Hybrid Search

Most RAG systems use only semantic search. This one uses both:
- **Semantic** (70%): "company earnings" matches "corporate revenue" by meaning
- **Keyword** (30%): "Q3 2024" matches chunks with exactly "Q3 2024"
- **RRF fusion**: Chunks that rank high in both get boosted

### Multimodal Understanding

PDFs contain more than text. This system extracts:
- **Text** — paragraphs, headings, body content
- **Tables** — converted to structured markdown for accurate number retrieval
- **Images** — charts, diagrams, and figures are described by GPT-4.1 mini vision, making them searchable

### Behind the Scenes (BTS) Toggle

The BTS panel shows you exactly what happened for each query — not AI commentary, but actual pipeline data: your query's embedding dimensions, search times in milliseconds, chunk relevance scores, how many candidates were found, and which sources were used.

---

## Project Structure

```
multimodal-rag/
├── app.py              # Streamlit UI (futuristic dark theme, streaming)
├── rag_chain.py        # Hybrid retrieval + cache + LLM + BTS generation
├── ingest.py           # PDF → text + tables + images → embeddings → Qdrant
├── semantic_cache.py   # Qdrant-based semantic similarity cache
├── config.py           # All settings in one place
├── setup.sh            # One-command installation
├── requirements.txt    # Python dependencies
├── .env.example        # API key template
├── .gitignore          # Keeps secrets out of git
└── data/               # Drop your PDFs here
```

---

## Configuration

Everything lives in `config.py`:

```python
# Models
EMBEDDING_MODEL = "text-embedding-3-small"    # Or "text-embedding-3-large"
LLM_MODEL = "gpt-4.1-mini"                   # Or "gpt-4.1" for max quality

# Chunking
CHUNK_SIZE = 1000                             # Characters per chunk
CHUNK_OVERLAP = 200                           # Overlap between chunks

# Search
TOP_K = 5                                     # Chunks per query
HYBRID_SEARCH = True                          # Dense + sparse
SEMANTIC_WEIGHT = 0.7                         # Semantic vs keyword ratio
KEYWORD_WEIGHT = 0.3

# Cache
CACHE_ENABLED = True
CACHE_SIMILARITY_THRESHOLD = 0.92             # Lower = more cache hits, less precise
```

---

## Cost

| Operation | Cost |
|-----------|------|
| Ingest 50-page PDF with 10 images | ~$0.03 |
| Each fresh query | ~$0.006 |
| Each cached query | **$0.000** |
| **100 queries on a 50-page doc** | **~$0.20** |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Set OPENAI_API_KEY" | Add your key to `.env` |
| No documents loaded | Put PDFs in `data/` and restart |
| Stale answers after updating PDFs | Click "Clear DB" in sidebar, restart |
| Qdrant errors | Delete `qdrant_data/` folder, restart |

---

## License

MIT — use it however you want.
