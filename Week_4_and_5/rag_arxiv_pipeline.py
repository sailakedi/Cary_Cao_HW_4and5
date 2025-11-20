"""
rag_arxiv_pipeline.py
---------------------

Builds a simple RAG-ready index for arXiv cs.CL papers:
 1) Download PDFs via arxiv API (50 papers by default)
 2) Extract raw text from PDFs using PyMuPDF (fitz)
 3) Chunk text into <= 512-token windows with overlap
 4) Compute dense embeddings for each chunk using sentence-transformers
 5) Store metadata and chunks in SQLite (with FTS5) and embeddings in FAISS (IndexIDMap)

Usage:
  python rag_arxiv_pipeline.py --build --out_dir ./cscl_data --n_papers 50
"""

import os
import io
import argparse
import sqlite3
import json
import math
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm

import arxiv
import fitz  # PyMuPDF
import numpy as np
import urllib.request

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

import faiss

# ------------------------
# Configuration / Defaults
# ------------------------
DEFAULT_MODEL_NAME = "all-mpnet-base-v2"  # sentence-transformers model for embeddings
TOKENIZER_NAME = "sentence-transformers/all-mpnet-base-v2"  # tokenizer just for token counting
MAX_CHUNK_TOKENS = 512
CHUNK_STRIDE = 64      # overlap in tokens when sliding window
EMBEDDING_DIM = 768    # mpnet-base typical dim is 768
SQLITE_DB = "arxiv_cscl.db"
FAISS_INDEX_FILE = "faiss_index.bin"
PDF_DIR_NAME = "pdfs"

# ------------------------
# Utilities
# ------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

# ------------------------
# 1) Download arXiv PDFs
# ------------------------
def fetch_arxiv_pdfs(query: str = "cat:cs.CL", max_results: int = 50, out_dir: str = "data/pdfs") -> List[Dict]:
    """
    Searches arXiv and downloads PDFs.
    Returns a list of metadata dicts: {doc_id, title, authors, year, keywords, pdf_path}
    """
    ensure_dir(out_dir)
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    downloaded = []
    print(f"[fetch_arxiv_pdfs] querying arXiv: {query}, max_results={max_results}")
    for result in tqdm(search.results(), total=max_results):
        arxiv_id = result.get_short_id()
        title = result.title
        authors = [a.name for a in result.authors]
        year = result.published.year if result.published else None

        pdf_filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_path = os.path.join(out_dir, pdf_filename)

        #url = result.pdf_url
        try:
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            urllib.request.urlretrieve(url, pdf_path)
        except:
            base_id = arxiv_id.split('v')[0]
            url = f"https://arxiv.org/pdf/{base_id}.pdf"
            urllib.request.urlretrieve(url, pdf_path)
        if url is None:
            print(f"⚠️  No PDF URL found for {arxiv_id} — skipping.")
            continue

        # download PDF if not present
        if not os.path.exists(pdf_path):
            try:
                print(f"Downloading {arxiv_id} → {pdf_path}")
                # manually download to avoid arxiv library join() issue
                with urllib.request.urlopen(url) as response, open(pdf_path, "wb") as f:
                    f.write(response.read())
                print("Done.")
            except Exception as e:
                print(f"Warning: failed to download {arxiv_id}: {e}")
                continue
        downloaded.append({
            "doc_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": year,
            "keywords": [],
            "pdf_path": pdf_path,
        })
    print(f"[fetch_arxiv_pdfs] downloaded {len(downloaded)} PDFs to {out_dir}")
    return downloaded

# ------------------------
# 2) Text Extraction
# ------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract full-document text by concatenating page-level get_text("text")
    """
    text_pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        page_text = page.get_text("text")
        # minimal cleaning
        page_text = page_text.replace("\x0c", "\n").strip()
        text_pages.append(page_text)
    doc.close()
    doc_text = "\n\n".join(p for p in text_pages if p)
    return doc_text

# ------------------------------
# 3) Extract Keywords Via TF-IDF
# ------------------------------
def extract_keywords_tfidf(docs, top_k=10):
    vectorizer = TfidfVectorizer(stop_words='english', min_df=2, max_df=0.85)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    keywords_per_doc = defaultdict(list)
    for i, row in enumerate(X.toarray()):
        top_indices = row.argsort()[-top_k:][::-1]
        keywords = [feature_names[idx] for idx in top_indices]
        keywords_per_doc[i] = keywords
    return keywords_per_doc

# ------------------------
# 4) Token-aware chunking
# ------------------------
def chunk_text_tokenwise(text: str,
                         tokenizer,
                         max_tokens: int = MAX_CHUNK_TOKENS,
                         stride: int = CHUNK_STRIDE) -> List[Dict]:
    """
    Create overlapping token-window chunks.
    Returns list of dicts: {chunk_text, token_start, token_end, n_tokens}
    """
    # encode text into token ids
    enc = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(enc)
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        token_window = enc[start:end]
        chunk_text = tokenizer.decode(token_window, skip_special_tokens=True)
        chunks.append({
            "chunk_text": chunk_text.strip(),
            "token_start": start,
            "token_end": end,
            "n_tokens": end - start
        })
        if end == total_tokens:
            break
        start = end - stride  # overlap
    return chunks

# A helper to split on section boundaries first
def split_on_section_headings(text: str) -> List[str]:
    """
    Very simple heuristic: split on common section headings (Introduction, Methods, Conclusion, References) 
    Return list of text blocks.
    """
    import re
    # pattern catches lines that are headings by themselves (e.g., "1 Introduction" or "Introduction")
    pattern = re.compile(r'(?m)^(?:\d+\s*)?(?:Abstract|Introduction|Related Work|Methods|Methodology|Approach|Experiments|Results|Discussion|Conclusion|Conclusions|References)\b.*$')
    # find headings indices
    splits = pattern.split(text)
    if len(splits) <= 1:
        return [text]
    # pattern.split returns content between matches, safer to use splitpoints approach.
    # For simplicity return the splits (they will include both headings and bodies).
    return [s.strip() for s in splits if s.strip()]

# ------------------------
# 5) Embeddings
# ------------------------
class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        print(f"[Embedder] loading model {model_name} ...")

        home_dir = os.environ.get('HOME')
        hf_catch_dir = "MyProjects/Week_4_and_5/hf_cache"
        if home_dir:
            os.environ["HF_HOME"] = os.path.join(home_dir, hf_catch_dir)
            if not os.path.exists(os.environ["HF_HOME"]):
               os.makedirs(os.environ["HF_HOME"], exist_ok=True)
        
        self.model = SentenceTransformer(model_name)
        # lock dimension from model
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[Embedder] model dimension: {self.dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Returns np.ndarray shape (n_texts, dim)
        """
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        return embeddings

# ------------------------
# 6) SQLite + FTS5 + FAISS management
# ------------------------
def init_sqlite_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # documents: metadata per paper
    cur.execute("""
    DROP TABLE IF EXISTS documents;
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        title TEXT,
        authors TEXT,
        year INTEGER,
        keywords TEXT,
        pdf_path TEXT
    )
    """)

    # chunks: stores chunk_text and references to doc
    cur.execute("""
    DROP TABLE IF EXISTS chunks;
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT,
        chunk_text TEXT,
        token_start INTEGER,
        token_end INTEGER,
        n_tokens INTEGER,
        page_range TEXT,
        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
    )
    """)
    # FTS5 table to allow textual search on chunks
    # using external contentless FTS table to avoid duplication isn't necessary here; keep simple:
    try:
        cur.execute("DROP TABLE IF EXISTS chunks_fts")
        cur.execute("""
                    CREATE VIRTUAL TABLE chunks_fts USING fts5(
                        chunk_id UNINDEXED,
                        doc_id UNINDEXED,
                        chunk_text,
                        tokenize = 'unicode61')
                    """)
    except sqlite3.OperationalError as e:
        # If SQLite does not have FTS5 enabled this will fail
        print("Warning: could not create FTS5 virtual table. Your Python SQLite build may not support FTS5.")
        print(e)
    conn.commit()
    print("SQLite database init....finished")
    return conn

def insert_document(conn: sqlite3.Connection, doc: Dict):
    cur = conn.cursor()
    cur.execute("""
    INSERT OR REPLACE INTO documents (doc_id, title, authors, year, keywords, pdf_path)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (doc["doc_id"], doc["title"], json.dumps(doc["authors"]), doc["year"], json.dumps(doc["keywords"]), doc["pdf_path"]))
    conn.commit()

def insert_chunk(conn: sqlite3.Connection, doc_id: str, chunk_text: str, token_start: int, token_end: int, n_tokens: int, page_range: str = None) -> int:
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO chunks (doc_id, chunk_text, token_start, token_end, n_tokens, page_range)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, chunk_text, token_start, token_end, n_tokens, page_range))
    chunk_id = cur.lastrowid
    # also insert into FTS5
    try:
        cur.execute("INSERT INTO chunks_fts(chunk_id, doc_id, chunk_text) VALUES (?, ?, ?)", (chunk_id, doc_id, chunk_text))
    except sqlite3.OperationalError:
        # ignore if FTS5 not enabled
        pass
    conn.commit()
    return chunk_id

# FAISS helper: create or load existing index, maintain mapping index id -> chunk_id
def init_faiss_index(dim: int, index_path: str = FAISS_INDEX_FILE) -> faiss.IndexIDMap:
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            print(f"[FAISS] creating new index (dim={dim})")
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(quantizer)  # mapping external ids
        except PermissionError:
            print(f"Permission denied: Unable to delete '{index_path}'.")
        except Exception as e:
            print(f"An error occurred while deleting '{index_path}': {e}")
    else:
        print(f"File '{index_path}' does not exist.")

    """
    if os.path.exists(index_path):
        print(f"[FAISS] loading index from {index_path}")
        index = faiss.read_index(index_path)
    else:
        print(f"[FAISS] creating new index (dim={dim})")
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(quantizer)  # mapping external ids
    """
    
    return index

def add_embeddings_to_faiss(index: faiss.IndexIDMap, embeddings: np.ndarray, chunk_ids: List[int], index_path: str = FAISS_INDEX_FILE):
    """
    embeddings shape (n, dim), chunk_ids list of ints same len
    """
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    ids = np.array(chunk_ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, index_path)
    print(f"[FAISS] added {len(chunk_ids)} vectors; saved index to {index_path}")

# ------------------------
# Putting all steps together
# ------------------------
def build_pipeline(out_dir: str, n_papers: int):
    ensure_dir(out_dir)
    pdf_dir = os.path.join(out_dir, PDF_DIR_NAME)
    ensure_dir(pdf_dir)

    # 1) fetch PDFs
    papers = fetch_arxiv_pdfs(query="cat:cs.CL", max_results=n_papers, out_dir=pdf_dir)

    # 2) extract text
    print("[extract] extracting text from PDFs ...")
    text_docs = []
    for p in tqdm(papers):
        if "text" not in p:
            try:
                p["text"] = extract_text_from_pdf(p["pdf_path"])
            except Exception as e:
                print(f"Failed to extract {p['pdf_path']}: {e}")
                p["text"] = ""
            text_docs.append(p["text"])
    print("[extract] extracting text from PDFs ... finished")

    # 3) generate keywords for each chunk
    #doc_text = "\n\n".join(p for p in text_pages if p)
    #keywords = extract_keywords_tfidf(text_docs)
    #for paper, kw in zip(papers, keywords):
    #    paper["keywords"] = kw

    # 4) prepping tokenizer for chunking
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)

    # 5) embedder
    embedder = Embedder(DEFAULT_MODEL_NAME)
    dim = embedder.dim

    # init DB and FAISS
    conn = init_sqlite_db(os.path.join(out_dir, SQLITE_DB))
    index = init_faiss_index(dim, os.path.join(out_dir, FAISS_INDEX_FILE))

    # Process each paper: insert metadata, chunk, compute embeddings per file and add to FAISS
    for paper in tqdm(papers, desc="Papers"):
        doc_id = paper["doc_id"]
        insert_document(conn, paper)

        text = paper.get("text", "")
        if not text.strip():
            continue

        # optional: split by section headings for more semantic chunks (fewer huge chunks)
        blocks = split_on_section_headings(text)
        #print(f"This paper has {len(blocks)} blocks")

        # gather all chunk_texts for this doc to embed in a batch
        all_chunks_for_doc = []
        chunk_metas = []  # will store (token_start, token_end, n_tokens)
        for block in blocks:
            block_chunks = chunk_text_tokenwise(block, tokenizer, max_tokens=MAX_CHUNK_TOKENS, stride=CHUNK_STRIDE)
            for ch in block_chunks:
                # Discard very short chunks (e.g., <= 5 tokens)
                if ch["n_tokens"] < 5:
                    continue
                all_chunks_for_doc.append(ch["chunk_text"])
                chunk_metas.append((ch["token_start"], ch["token_end"], ch["n_tokens"]))

        if not all_chunks_for_doc:
            continue

        # compute embeddings in batches
        embeddings = embedder.embed_texts(all_chunks_for_doc, batch_size=32)  # shape (n_chunks, dim)

        # insert chunks to sqlite and add to FAISS
        chunk_ids = []
        for i, chunk_text in enumerate(all_chunks_for_doc):
            token_start, token_end, n_tokens = chunk_metas[i]
            chunk_id = insert_chunk(conn, doc_id, chunk_text, token_start, token_end, n_tokens)
            chunk_ids.append(chunk_id)

        # add embeddings with external ids equal to chunk_ids
        add_embeddings_to_faiss(index, embeddings, chunk_ids, index_path=os.path.join(out_dir, FAISS_INDEX_FILE))

    cur = conn.cursor()
    cur.execute("select count(*) from documents")
    row = cur.fetchone()
    print(f"Total {row[0]} records in database DOCUMENTS")

    cur.execute("select count(*) from chunks")
    row = cur.fetchone()
    print(f"Total {row[0]} records in database CHUNKS")

    cur.execute("select count(*) from chunks_fts")
    row = cur.fetchone()
    print(f"Total {row[0]} records in database CHUNKS_FTS")

    print("The number of FAISS vectors:", index.ntotal)

    conn.close()
    print("[Done] pipeline built.")

# ------------------------
# Retrieval + simple RAG demo (k-NN retrieval then simple concatenation)
# ------------------------
def retrieve_similar_chunks(query: str, out_dir: str, k: int = 5) -> List[Tuple[int, float, str]]:
    """
    Return list of tuples (chunk_id, distance, chunk_text)
    """
    # load resources
    conn = sqlite3.connect(os.path.join(out_dir, SQLITE_DB))
    cur = conn.cursor()

    embedder = Embedder(DEFAULT_MODEL_NAME)
    index = init_faiss_index(embedder.dim, os.path.join(out_dir, FAISS_INDEX_FILE))

    q_emb = embedder.embed_texts([query])[0].astype(np.float32)
    D, I = index.search(np.expand_dims(q_emb, axis=0), k)
    ids = I[0]
    dists = D[0]
    results = []
    for idx, dist in zip(ids, dists):
        if idx == -1:
            continue
        cur.execute("SELECT chunk_text FROM chunks WHERE chunk_id = ?", (int(idx),))
        row = cur.fetchone()
        if row:
            results.append((int(idx), float(dist), row[0]))
    conn.close()
    return results

def rag_answer(query: str, out_dir: str, k: int = 5) -> str:
    """
    Very simple RAG: retrieve top-k chunks, concatenate, and return as context.
    You can feed the returned context + question to any LLM for generation.
    """
    hits = retrieve_similar_chunks(query, out_dir, k=k)
    context = "\n\n---\n\n".join(h[2] for h in hits)
    return context

# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data", help="output directory")
    parser.add_argument("--n_papers", type=int, default=50, help="number of arXiv papers to download")
    parser.add_argument("--build", action="store_true", help="build index")
    parser.add_argument("--query", type=str, default=None, help="run a simple retrieval query against built index")
    parser.add_argument("--k", type=int, default=5, help="number of neighbors to retrieve")
    args = parser.parse_args()

    if args.build:
        build_pipeline(args.out_dir, args.n_papers)

    if args.query:
        print("[RAG Demo] retrieving top-k chunks")
        context = rag_answer(args.query, args.out_dir, k=args.k)
        print("==== Retrieved context ====\n")
        print(context[:2000])  # print first 2000 chars
        print("\n\n==== End context ====")
        print("\nNow you can pass 'context' + 'question' to your favorite LLM to generate the final answer.")

if __name__ == "__main__":
    main()
