"""
ingest.py — The ingestion pipeline.

Drop PDFs into data/ → run this script (or let app.py auto-ingest).
Extracts text + images + tables → describes images → chunks → embeds → Qdrant.

Usage:
    python ingest.py                   # Ingest all PDFs in data/
    python ingest.py path/to/file.pdf  # Ingest a single PDF
"""

import re
import sys
import base64
import hashlib
import time
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVector, SparseVectorParams, SparseIndexParams, VectorParamsMap,
)

import config

# ── Clients ──────────────────────────────────────────
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
qdrant_client = QdrantClient(path=str(config.QDRANT_PATH))
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY,
)


# ─────────────────────────────────────────────────────
# STEP 1 — Extract text
# ─────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# ─────────────────────────────────────────────────────
# STEP 1b — Extract tables as markdown
# ─────────────────────────────────────────────────────
def extract_tables(pdf_path: str) -> list[dict]:
    if not config.EXTRACT_TABLES:
        return []

    doc = fitz.open(pdf_path)
    tables = []
    for i, page in enumerate(doc):
        try:
            for table in page.find_tables():
                df = table.to_pandas()
                if df.empty or df.shape[0] < 2:
                    continue
                headers = [str(h).strip() for h in df.columns]
                md_lines = ["| " + " | ".join(headers) + " |"]
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for _, row in df.iterrows():
                    md_lines.append("| " + " | ".join(str(c).strip() for c in row) + " |")
                tables.append({"page": i + 1, "text": f"[TABLE]\n" + "\n".join(md_lines)})
        except Exception:
            continue
    doc.close()
    return tables


# ─────────────────────────────────────────────────────
# STEP 2 — Extract images
# ─────────────────────────────────────────────────────
def extract_images(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    images = []
    for i, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.width < config.MIN_IMAGE_SIZE or pix.height < config.MIN_IMAGE_SIZE:
                    pix = None
                    continue
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_hash = hashlib.md5(pix.samples).hexdigest()[:10]
                img_path = config.IMAGE_CACHE / f"page{i+1}_img{img_index}_{img_hash}.png"
                pix.save(str(img_path))

                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")

                images.append({"page": i + 1, "image_path": str(img_path), "image_base64": b64})
                pix = None
            except Exception as e:
                print(f"    ⚠️  Skipped image on page {i+1}: {e}")
    doc.close()
    return images


# ─────────────────────────────────────────────────────
# STEP 3 — Describe images with vision
# ─────────────────────────────────────────────────────
def describe_image(image_base64: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model=config.LLM_MODEL,
                max_tokens=config.MAX_IMAGE_DESCRIBE_TOKENS,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (
                            "Describe this image in detail for a document retrieval system. "
                            "Include: what it shows, any text/labels, data values if chart/table, "
                            "and the key takeaway. Be precise with all numbers and data points."
                        )},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{image_base64}", "detail": "high"
                        }},
                    ],
                }],
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "[Image could not be described — API error]"


# ─────────────────────────────────────────────────────
# STEP 4 — Chunking
# ─────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(pages, source):
    chunks = []
    for p in pages:
        for j, text in enumerate(splitter.split_text(p["text"])):
            chunks.append({"text": text, "metadata": {
                "source": source, "page": p["page"], "chunk_index": j, "content_type": "text",
            }})
    return chunks


def chunk_tables(tables, source):
    return [{"text": t["text"], "metadata": {
        "source": source, "page": t["page"], "chunk_index": i, "content_type": "table",
    }} for i, t in enumerate(tables)]


def chunk_images(images, source):
    chunks = []
    for img in images:
        print(f"    🔍 Describing image on page {img['page']}...")
        desc = describe_image(img["image_base64"])
        chunks.append({"text": desc, "metadata": {
            "source": source, "page": img["page"],
            "content_type": "image", "image_path": img["image_path"],
        }})
    return chunks


# ─────────────────────────────────────────────────────
# STEP 5 — Sparse vectors (BM25-like)
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
# STEP 6 — Store in Qdrant
# ─────────────────────────────────────────────────────
def ensure_collection():
    collections = [c.name for c in qdrant_client.get_collections().collections]
    if config.QDRANT_COLLECTION not in collections:
        sparse_config = None
        if config.HYBRID_SEARCH:
            sparse_config = {"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}

        qdrant_client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParamsMap(map={
                "dense": VectorParams(size=config.EMBEDDING_DIMS, distance=Distance.COSINE),
            }),
            sparse_vectors_config=sparse_config,
        )
        print(f"  ✅ Created collection: {config.QDRANT_COLLECTION}")


def store_chunks(chunks):
    if not chunks:
        return
    texts = [c["text"] for c in chunks]
    print(f"  📐 Embedding {len(texts)} chunks...")
    dense_vectors = embeddings.embed_documents(texts)

    try:
        id_offset = qdrant_client.get_collection(config.QDRANT_COLLECTION).points_count
    except Exception:
        id_offset = 0

    points = []
    for i, (chunk, dvec) in enumerate(zip(chunks, dense_vectors)):
        vec = {"dense": dvec}
        if config.HYBRID_SEARCH:
            vec["sparse"] = compute_sparse_vector(chunk["text"])
        points.append(PointStruct(id=id_offset + i, vector=vec, payload={"text": chunk["text"], **chunk["metadata"]}))

    for start in range(0, len(points), 100):
        qdrant_client.upsert(collection_name=config.QDRANT_COLLECTION, points=points[start:start+100])
    print(f"  💾 Stored {len(points)} chunks in Qdrant")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def ingest_pdf(pdf_path: str) -> int:
    source = Path(pdf_path).name
    print(f"\n📄 Processing: {source}")

    pages = extract_text(pdf_path)
    print(f"  📝 {len(pages)} pages with text")
    tables = extract_tables(pdf_path)
    print(f"  📊 {len(tables)} tables")
    images = extract_images(pdf_path)
    print(f"  🖼️  {len(images)} images")

    text_chunks = chunk_text(pages, source)
    table_chunks = chunk_tables(tables, source)
    image_chunks = chunk_images(images, source)
    all_chunks = text_chunks + table_chunks + image_chunks
    print(f"  ✂️  {len(text_chunks)} text + {len(table_chunks)} table + {len(image_chunks)} image = {len(all_chunks)} chunks")

    ensure_collection()
    store_chunks(all_chunks)
    print(f"  ✅ Done: {source}\n")
    return len(all_chunks)


def ingest_directory(dir_path=None):
    """Ingest all PDFs in a directory. Skips if already ingested."""
    directory = Path(dir_path) if dir_path else config.DATA_DIR
    pdfs = list(directory.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  No PDFs found in {directory}/")
        return 0

    # Check if already ingested
    try:
        info = qdrant_client.get_collection(config.QDRANT_COLLECTION)
        if info.points_count > 0:
            print(f"✅ Already ingested ({info.points_count} chunks in DB). Skipping.")
            return info.points_count
    except Exception:
        pass

    total = 0
    for pdf in pdfs:
        total += ingest_pdf(str(pdf))
    print(f"🎉 Ingestion complete! {total} chunks from {len(pdfs)} PDF(s)")
    return total


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])
    else:
        ingest_directory()
