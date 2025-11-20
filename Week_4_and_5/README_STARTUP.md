# 1. rag_arxiv_pipeline.py
#    ---------------------
#
#    Builds a simple RAG-ready index for arXiv cs.CL papers:
#       1) Download PDFs via arxiv API (50 papers by default)
#       2) Extract raw text from PDFs using PyMuPDF (fitz)
#       3) Chunk text into <= 512-token windows with overlap
#       4) Compute dense embeddings for each chunk using sentence-transformers
#       5) Store metadata and chunks in SQLite (with FTS5) and embeddings in FAISS (IndexIDMap)
#
#    The Structure of This Program
#    Project
#    |--> rag_arxiv_pipeline.py
#    |--> cscl_data
#    |    |--> arxiv_cscl.db (A SQLite database storing metadata and chunks)
#    |    |--> faiss_index.bin (A index file storing faiss indeies)
#    |    |--> pdfs
#    |    |    |--> xxx.pdf (50 pdf files downloaded from arXiv cs.CL papers)
#    |    |         ...
#    |--> hf_cache
#    |    |--> ... (some cache files used by this program)
#
#    Usage:
       python rag_arxiv_pipeline.py --build --out_dir ./cscl_data --n_papers 50

# 2. hybrid_search_app
#    -----------------
# 
#    FastAPI app for hybrid semantic, keyword and hybrid search from thunks.
#
#    The Sturcture of This Program
#    Project
#    |--> rhybrid_search_app       
#    |--> index.html 
#    |--> screenshot_for_compairing_three_searches.png
#    |--> __pycache__
#         |--> ... (some cache files used by this program)
#
#    Usage:
#       1) Start up Backend
           cd ~/Week_4_and_5
           uvicorn hybrid_search_app:app --reload
#       2) Start up Frontend from browser
           http://127.0.0.1:8000/

# 3. evaluate_hybrid_app
#    -------------------
# 
#    An evaluation program showing at least 10 example queries and reporting 
#    hit_rate_@3 for semantic-only, keyword-only, and hybrid search.
#
#    The Sturcture of This Program
#    Project
#    |--> evaluate_hybrid_app.py
#    |--> evaluate_hybrid.py    
#    |--> screenshot_for_evaluation_of_three_searches.png
#    |--> templates
#    |    |--> results.html
#    |--> __pycache__
#         |--> ... (some cache files used by this program)
#
#    Usage:
#       1) Start up Backend
           cd ~/Week_4_and_5
           python evaluate_hybrid.py
#       2) Start up Frontend from browser
           http://127.0.0.1:8000/results