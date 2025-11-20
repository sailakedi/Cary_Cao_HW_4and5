"""
evaluate_hybrid.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#import sqlite3
#import requests
from hybrid_search_app import faiss_search, fts_search, reciprocal_rank_fusion

# ---------------------------------------------
# Configuration
# ---------------------------------------------
API = "http://127.0.0.1:8000"  # optional if you want to call real endpoints
TOP_K = 3

# Example gold standard: queries mapped to relevant doc_ids
TEST_QUERIES = {
    "transformer summarization": ["2511.10618v1"],
    "machine translation evaluation": ["2511.10338v1"],
    "dialogue response generation": ["2511.10215v1"],
    "multilingual embedding alignment": ["2511.10229v1"],
    "BERT fine-tuning stability": ["2511.10051v1"],
    "speech recognition end-to-end": ["2511.10338v1"],
    "text simplification": ["2511.10281v1"],
    "contrastive learning NLP": ["2511.10400v1"],
    "semantic parsing": ["2511.10192v1"],
    "knowledge distillation": ["2511.10093v1"]
}

# ---------------------------------------------
# Dummy search function
# ---------------------------------------------
def search_dummy(fn_name: str, query):
    k = TOP_K
    doc_ids = []

    if fn_name == "semantic":
        results =  faiss_search(query, k) 
    elif fn_name == "keyworks":
        results = fts_search(query, k)
    else:
        results = reciprocal_rank_fusion(faiss_search(query, k),
                                         fts_search(query, k),
                                         k)

    doc_ids = [item.get("doc_id") for item in results]
    doc_ids = list(dict.fromkeys(doc_ids))
    return doc_ids[:3]

# ---------------------------------------------
# Helper: compute hit at k
# ---------------------------------------------
def hit_at_k(result_ids, relevant_ids, k=TOP_K):
    top = result_ids[:k]
    return 1 if any(doc_id in relevant_ids for doc_id in top) else 0

# ---------------------------------------------
# Main evaluation function
# ---------------------------------------------
def evaluate(k=TOP_K):
    """
    Returns a dictionary formatted for the Jinja2 template.
    {
        "rows": [
            {"query": "...", 
             "verified": [...], 
             "semantic": [...], 
             "keyword": [...], 
             "hybrid": [...]},
            ...
        ],
        "hit_semantic": 0.67,
        "hit_keyword": 0.53,
        "hit_hybrid": 0.80
    }
    """
    rows = []
    semantic_hits = []
    keywords_hits = []
    hybrid_hits = []

    for query, relevant_docs in TEST_QUERIES.items():
        semantic_docs = search_dummy("semantic", query)
        keywords_docs = search_dummy("keywords", query)
        hybrid_docs = search_dummy("hybrid", query)

        rows.append({
            "query": query,
            "verified": relevant_docs,
            "semantic": semantic_docs,
            "keyword": keywords_docs,
            "hybrid": hybrid_docs
        })

        semantic_hits.append(hit_at_k(semantic_docs, relevant_docs, k))
        keywords_hits.append(hit_at_k(keywords_docs, relevant_docs, k))
        hybrid_hits.append(hit_at_k(hybrid_docs, relevant_docs, k))

    n_queries = len(TEST_QUERIES)
    return {
        "rows": rows,
        "hit_semantic": round(sum(semantic_hits)/n_queries, 3),
        "hit_keyword": round(sum(keywords_hits)/n_queries, 3),
        "hit_hybrid": round(sum(hybrid_hits)/n_queries, 3)
    }

# ---------------------------------------------
# Optional: run standalone
# ---------------------------------------------
if __name__ == "__main__":
    results = evaluate(k=3)
    import json
    print(json.dumps(str(results), indent=2))
