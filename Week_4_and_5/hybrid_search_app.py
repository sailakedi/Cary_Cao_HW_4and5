"""
hybrid_search_app.py
--------------------

FastAPI app for hybrid semantic + keyword search on arXiv cs.CL papers.

Usage:
    uvicorn hybrid_search_app:app --reload
Then test:
    http://localhost:8000/
"""
import os
import re
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from fastapi.responses import FileResponse

# -------------------
# Configuration
# -------------------
MODEL_NAME = "all-mpnet-base-v2"
FAISS_INDEX_FILE = "faiss_index.bin"
DB_FILE = "arxiv_cscl.db"
RRF_K = 60  # reciprocal rank fusion constant
TOP_K = 5

app = FastAPI(title="Hybrid Search API (FAISS + FTS5)")

# -------------------
# Load models and indexes
# -------------------
print("[Init] Loading SentenceTransformer model...")
embedder = SentenceTransformer(MODEL_NAME)

home_dir = os.environ.get('HOME')
data_dir = "MyProjects/Week_4_and_5/cscl_data"
if home_dir:
    os.environ["DATA_HOME"] = os.path.join(home_dir, data_dir)
    faiss_index_path = os.path.join(os.environ["DATA_HOME"], FAISS_INDEX_FILE)
    db_path = os.path.join(os.environ["DATA_HOME"], DB_FILE)

print("[Init] Loading FAISS index...")
faiss_index = faiss.read_index(faiss_index_path)
dim = faiss_index.d
print(f"[Init] FAISS index loaded (dim={dim})")

def normalize_query(q: str, is_keywords_search: bool) -> str:
    ## replace curly quotes
    q = q.replace("“", '"').replace("”", '"')
    q = q.replace("‘", "'").replace("’", "'")

    # lowercase
    q = q.lower()

    if (is_keywords_search):
        q = q.replace("-", " ")

        # remove unwanted characters except alphanumerics and space
        q = re.sub(r"[^\w\s]", " ", q)

        # collapse spaces
        q = re.sub(r'\s+', ' ', q).strip()

        tokens = q.split()
        q = " OR ".join(tokens)

    return q

# -------------------
# FAISS semantic search
# -------------------
def faiss_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:

    # NEW: clean the query for FTS5
    query = normalize_query(query, is_keywords_search = False)
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    D, I = faiss_index.search(q_emb, top_k)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row 
    cur = conn.cursor()
    results = []
    
    for rank, (chunk_id, dist) in enumerate(zip(I[0][::-1], D[0][::-1])):
        if chunk_id == -1:
            continue
        cur.execute("""
            SELECT d.title, d.authors, c.doc_id, c.chunk_id, c.chunk_text
            FROM chunks c 
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ?
        """, (int(chunk_id),))

        row = cur.fetchone()
        if row and rank < top_k:
            results.append({
                "title": row["title"],
                "authors": row["authors"],
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "chunk_text": row["chunk_text"],
                "score": float(dist),
                "rank": rank + 1,
                "source": "semantic"
            })
    conn.close()
    return results

# -------------------
# Keyword search via FTS5
# -------------------
def fts_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query = normalize_query(query, is_keywords_search = True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Using match operator for FTS5
    try:
        cur.execute("""
            SELECT d.title, d.authors, c.doc_id, c.chunk_id, c.chunk_text, bm25(chunks_fts) AS score
            FROM chunks_fts 
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
        """, (query, top_k))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        # fallback: simple LIKE if FTS5 unavailable
        cur.execute("""
            SELECT d.title, d.authors, c.doc_id, c.chunk_id, c.chunk_text, 0.0 as score
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_text LIKE ?
            LIMIT ?
        """, (f"%{query}%", top_k))
        rows = cur.fetchall()

    results = []
    for rank, row in enumerate(rows):
        if rank < top_k:
            results.append({
                "title": row["title"],
                "authors": row["authors"],
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "chunk_text": row["chunk_text"],
                "score": float(row["score"]),
                "rank": rank + 1,
                "source": "keyword"
            })
    return results

# -------------------
# Score fusion via Reciprocal Rank Fusion (RRF)
# -------------------
def reciprocal_rank_fusion(semantic: List[Dict], keyword: List[Dict], k: int = 60, top_k: int = 10):
    """Combine ranked lists using RRF."""
    combined = {}
    for lst in [semantic, keyword]:
        for r in lst:
            cid = r["chunk_id"]
            rank = r["rank"]
            score = 1.0 / (k + rank)
            combined[cid] = combined.get(cid, 0) + score

    # fetch chunk info again for final results
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    results = []
    rank = 0

    for cid, fused_score in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        cur.execute("""
            SELECT d.title, d.authors, c.doc_id, c.chunk_id, c.chunk_text
            FROM chunks c 
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ?
        """, (cid,))
        row = cur.fetchone()

        if row and rank < top_k:
            rank = rank + 1
            results.append({
                "title": row["title"],
                "authors": row["authors"],
                "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"],
                "chunk_text": row["chunk_text"],
                "fused_score": fused_score,
                "rank": rank,
                "source": "hybrid"
            })
    conn.close()
    return results

# -------------------
# FastAPI Endpoint
# -------------------
@app.get("/hybrid_search")
async def hybrid_search(
    q: str = Query(..., description="User query string"),
    k: int = Query(3, description="Number of results to return")
):
    # 1. Semantic (FAISS)
    semantic_results = faiss_search(q, top_k=k)
    # 2. Keyword (FTS5)
    keyword_results = fts_search(q, top_k=k)
    # 3. Fusion (RRF)
    hybrid_results = reciprocal_rank_fusion(semantic_results, keyword_results, top_k=k)
    # Normalize format
    response = {
        "query": q,
        "semantic_results": semantic_results,
        "keyword_results": keyword_results,
        "hybrid_results": hybrid_results
    }
    return JSONResponse(content=response)

@app.get("/")
async def root():
    return FileResponse("index.html")
