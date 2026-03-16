"""
app.py — Futuristic Chat UI for Multimodal RAG.

Features:
  - Auto-ingests PDFs from data/ on startup (no upload needed)
  - Semantic cache for repeated/similar questions
  - "Behind the Scenes" toggle explaining the full RAG pipeline
  - Streaming responses with 🧑 / 🤖 avatars
  - Curved floating input bar

Run:  streamlit run app.py
"""

import streamlit as st
import config

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Bing — Multimodal RAG",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════
# FUTURISTIC CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ────────────────────────────── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: #1a1a2e;
    --bg-glass: rgba(26, 26, 46, 0.6);
    --accent-cyan: #00d4ff;
    --accent-purple: #7b5ea7;
    --accent-glow: rgba(0, 212, 255, 0.15);
    --text-primary: #e8e8f0;
    --text-secondary: #8888a0;
    --text-muted: #555570;
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-glow: rgba(0, 212, 255, 0.2);
    --font-main: 'Outfit', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Override Streamlit background */
.stApp, [data-testid="stAppViewContainer"], .main, section.main {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

.block-container {
    max-width: 820px !important;
    padding-top: 1rem !important;
    padding-bottom: 6rem !important;
}

/* ── Header ────────────────────────────── */
.hero-title {
    text-align: center;
    padding: 2rem 0 0.5rem;
    user-select: none;
}
.hero-title h1 {
    font-family: var(--font-main);
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-cyan), #a78bfa, var(--accent-cyan));
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease-in-out infinite;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.92rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-bottom: 1.5rem;
}
@keyframes shimmer {
    0%,100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* ── Chat messages ─────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.6rem 0 !important;
    font-family: var(--font-main) !important;
}

/* User messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    /* no special bg for user */
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.08), rgba(123, 94, 167, 0.08));
    border: 1px solid var(--border-glow);
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    color: var(--text-primary);
}

/* Assistant messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px 20px 20px 4px;
    padding: 12px 18px;
    color: var(--text-primary);
}

/* ── Chat input — curved floating bar ──── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 1.2rem !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(92%, 780px) !important;
    z-index: 999 !important;
}
[data-testid="stChatInput"] > div {
    background: var(--bg-glass) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 50px !important;
    padding: 4px 8px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 60px rgba(0, 212, 255, 0.06) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    font-size: 0.95rem !important;
    caret-color: var(--accent-cyan) !important;
    background: transparent !important;
}
[data-testid="stChatInput"] button {
    background: var(--accent-cyan) !important;
    border-radius: 50% !important;
    color: #0a0a0f !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── Source badges ──────────────────────── */
.source-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.src-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.74rem;
    color: var(--accent-cyan);
    font-family: var(--font-mono);
    letter-spacing: 0.3px;
}
.src-badge.table { color: #4ade80; border-color: rgba(74, 222, 128, 0.2); background: rgba(74, 222, 128, 0.06); }
.src-badge.image { color: #facc15; border-color: rgba(250, 204, 21, 0.2); background: rgba(250, 204, 21, 0.06); }

/* ── Cache hit badge ───────────────────── */
.cache-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(250, 204, 21, 0.08);
    border: 1px solid rgba(250, 204, 21, 0.2);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #facc15;
    font-family: var(--font-mono);
    margin-bottom: 8px;
}

/* ── BTS panel ─────────────────────────── */
.bts-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 20px 24px;
    margin-top: 12px;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-secondary);
}
.bts-panel h3 {
    color: var(--accent-cyan);
    font-family: var(--font-main);
    font-size: 1rem;
    font-weight: 600;
    margin: 16px 0 6px;
}
.bts-panel h3:first-child { margin-top: 0; }
.bts-panel strong { color: var(--text-primary); }
.bts-panel code {
    background: rgba(0, 212, 255, 0.08);
    color: var(--accent-cyan);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 0.82rem;
}

/* ── Rewrite note ──────────────────────── */
.rewrite-note {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
    padding: 2px 0 6px;
    font-family: var(--font-mono);
}

/* ── Sidebar ───────────────────────────── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
}

/* ── Stats pill row ────────────────────── */
.stats-row {
    display: flex;
    gap: 8px;
    margin: 12px 0;
}
.stat-pill {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 10px 8px;
    text-align: center;
}
.stat-pill .num {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent-cyan);
    font-family: var(--font-mono);
}
.stat-pill .lbl {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ── Spinner ───────────────────────────── */
[data-testid="stSpinner"] { color: var(--accent-cyan) !important; }

/* ── Scrollbar ─────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 3px; }

/* ── Status container ──────────────────── */
[data-testid="stStatus"] {
    background: var(--bg-card) !important;
    border-color: var(--border-subtle) !important;
}

/* ── Markdown in dark ──────────────────── */
.stMarkdownContainer p, .stMarkdownContainer li {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False


# ══════════════════════════════════════════════════════
# AUTO-INGEST ON STARTUP
# ══════════════════════════════════════════════════════
if not st.session_state.ingested and config.AUTO_INGEST_ON_STARTUP:
    if not config.OPENAI_API_KEY:
        st.error("⚠️ Set `OPENAI_API_KEY` in your `.env` file to get started.")
        st.stop()

    from pathlib import Path
    pdfs = list(config.DATA_DIR.glob("*.pdf"))
    if pdfs:
        # Check if already ingested
        try:
            from qdrant_client import QdrantClient
            qc = QdrantClient(path=str(config.QDRANT_PATH))
            info = qc.get_collection(config.QDRANT_COLLECTION)
            if info.points_count > 0:
                st.session_state.ingested = True
        except Exception:
            pass

        if not st.session_state.ingested:
            with st.status("🔮 Preparing knowledge base...", expanded=True) as status:
                from ingest import ingest_pdf
                total = 0
                for i, pdf in enumerate(pdfs):
                    status.update(label=f"Processing {pdf.name}... ({i+1}/{len(pdfs)})")
                    st.write(f"📄 {pdf.name}")
                    n = ingest_pdf(str(pdf))
                    total += n
                    st.write(f"  ✅ {n} chunks")
                status.update(label=f"✅ Knowledge base ready — {total} chunks from {len(pdfs)} PDF(s)", state="complete", expanded=False)
            st.session_state.ingested = True
    else:
        st.session_state.ingested = True  # No PDFs to ingest


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    # BTS Toggle
    st.markdown("---")
    bts_enabled = st.toggle(
        "🚀 Curious about BTS? Click to explore!",
        value=False,
        help="See exactly how each query is processed — chunks, embeddings, search scores, and more",
    )
    if bts_enabled:
        st.caption("Each answer will include a detailed breakdown of the RAG pipeline — powered by real numbers, not AI.")

    st.markdown("---")

    # Stats
    st.markdown("### 📊 Knowledge base")
    try:
        from qdrant_client import QdrantClient
        qc = QdrantClient(path=str(config.QDRANT_PATH))
        info = qc.get_collection(config.QDRANT_COLLECTION)
        chunk_count = info.points_count

        # Get cache stats
        from semantic_cache import get_cache_stats
        cache_stats = get_cache_stats()

        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-pill"><div class="num">{chunk_count}</div><div class="lbl">Chunks</div></div>
            <div class="stat-pill"><div class="num">{cache_stats['cached_queries']}</div><div class="lbl">Cached</div></div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.info("No documents loaded yet")

    st.markdown("---")
    top_k = st.slider("Retrieval depth (Top-K)", 1, 15, config.TOP_K)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear DB", use_container_width=True):
            try:
                from qdrant_client import QdrantClient
                qc = QdrantClient(path=str(config.QDRANT_PATH))
                qc.delete_collection(config.QDRANT_COLLECTION)
                from semantic_cache import clear_cache
                clear_cache()
                st.session_state.ingested = False
                st.toast("Cleared!", icon="🗑️")
                st.rerun()
            except Exception:
                st.toast("Nothing to clear")
    with col2:
        if st.button("💬 New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hero-title"><h1>Hi, I'm Bing!</h1></div>
<div class="hero-subtitle">Your multimodal document brain — text, tables, charts, and images</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def render_sources(sources):
    if not sources:
        return
    badges = ""
    for s in sources:
        ctype = s.get("type", "text")
        cls = f"src-badge {ctype}" if ctype in ("image", "table") else "src-badge"
        icon = {"image": "🖼️", "table": "📊", "text": "📄"}.get(ctype, "📄")
        badges += f'<span class="{cls}">{icon} {s["source"]} p.{s["page"]} ({s["score"]})</span>'
    st.markdown(f'<div class="source-row">{badges}</div>', unsafe_allow_html=True)


def render_bts(bts_text):
    if not bts_text:
        return
    st.markdown(f'<div class="bts-panel">{_md_to_html(bts_text)}</div>', unsafe_allow_html=True)


def _md_to_html(md: str) -> str:
    """Minimal markdown→html for BTS panel (no external deps)."""
    import re
    html = md
    # Headers
    html = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    # Bold
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    # Inline code
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    # Line breaks
    html = html.replace("\n\n", "<br><br>")
    html = html.replace("\n", "<br>")
    return html


# ══════════════════════════════════════════════════════
# CHAT HISTORY DISPLAY
# ══════════════════════════════════════════════════════
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("rewritten_query"):
            st.markdown(f'<div class="rewrite-note">🔄 Searched: "{msg["rewritten_query"]}"</div>', unsafe_allow_html=True)
        if msg.get("cache_hit"):
            st.markdown(f'<span class="cache-badge">⚡ Cache hit — {msg["cache_similarity"]:.1%} similar</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if "sources" in msg:
            render_sources(msg["sources"])
        if msg.get("bts_text"):
            render_bts(msg["bts_text"])


# ══════════════════════════════════════════════════════
# CHAT INPUT + STREAMING RESPONSE
# ══════════════════════════════════════════════════════
if prompt := st.chat_input("Ask me anything..."):
    # ── User message ─────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    # ── Assistant response (streaming) ───────────
    with st.chat_message("assistant", avatar="🤖"):
        from rag_chain import ask_stream

        sources = []
        rewritten = None
        bts_text = None
        cache_hit = False
        cache_similarity = 0
        full_text = ""

        rewrite_ph = st.empty()
        cache_ph = st.empty()
        text_ph = st.empty()
        source_ph = st.empty()
        bts_ph = st.empty()

        for event in ask_stream(
            prompt,
            top_k=top_k,
            chat_history=st.session_state.chat_history,
            generate_bts=bts_enabled,
        ):
            etype = event["type"]

            if etype == "rewritten":
                rewritten = event["data"]
                rewrite_ph.markdown(
                    f'<div class="rewrite-note">🔄 Searched: "{rewritten}"</div>',
                    unsafe_allow_html=True,
                )

            elif etype == "cache_hit":
                d = event["data"]
                cache_hit = True
                cache_similarity = d["similarity"]
                sources = d["sources"]
                full_text = d["answer"]

                cache_ph.markdown(
                    f'<span class="cache-badge">⚡ Cache hit — {cache_similarity:.1%} similar</span>',
                    unsafe_allow_html=True,
                )
                text_ph.markdown(full_text)
                render_html = ""
                for s in sources:
                    ctype = s.get("type", "text")
                    cls = f"src-badge {ctype}" if ctype in ("image", "table") else "src-badge"
                    icon = {"image": "🖼️", "table": "📊", "text": "📄"}.get(ctype, "📄")
                    render_html += f'<span class="{cls}">{icon} {s["source"]} p.{s["page"]} ({s["score"]})</span>'
                if render_html:
                    source_ph.markdown(f'<div class="source-row">{render_html}</div>', unsafe_allow_html=True)

            elif etype == "sources":
                sources = event["data"]

            elif etype == "token":
                full_text += event["data"]
                text_ph.markdown(full_text + " ▌")

            elif etype == "bts":
                bts_text = event["data"]
                render_bts_in = bts_ph
                render_bts_in.markdown(
                    f'<div class="bts-panel">{_md_to_html(bts_text)}</div>',
                    unsafe_allow_html=True,
                )

            elif etype == "done":
                if not cache_hit:
                    text_ph.markdown(full_text)
                    # Render sources
                    render_html = ""
                    for s in sources:
                        ctype = s.get("type", "text")
                        cls = f"src-badge {ctype}" if ctype in ("image", "table") else "src-badge"
                        icon = {"image": "🖼️", "table": "📊", "text": "📄"}.get(ctype, "📄")
                        render_html += f'<span class="{cls}">{icon} {s["source"]} p.{s["page"]} ({s["score"]})</span>'
                    if render_html:
                        source_ph.markdown(f'<div class="source-row">{render_html}</div>', unsafe_allow_html=True)

        # ── Save to session state ────────────────
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": sources,
            "rewritten_query": rewritten,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
            "bts_text": bts_text,
        })

        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": full_text})

        max_entries = config.MAX_HISTORY_TURNS * 2
        if len(st.session_state.chat_history) > max_entries:
            st.session_state.chat_history = st.session_state.chat_history[-max_entries:]
