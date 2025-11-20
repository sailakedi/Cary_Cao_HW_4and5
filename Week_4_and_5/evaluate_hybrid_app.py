"""
evaluate_hybrid_app.py
----------------------

An evaluation program showing at least 10 example queries and reporting 
hit_rate_@3 for semantic-only, keyword-only, and hybrid search.

Usage:
    python evaluate_hybrid.py
Then test:
    http://127.0.0.1:8000/results
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from evaluate_hybrid import evaluate  # your existing evaluation function

app = FastAPI()

# ---------------------------------------------
# Mount static folder safely
# ---------------------------------------------
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"Warning: Static folder '{static_dir}' does not exist. Skipping static mount.")

# ---------------------------------------------
# Templates folder
# ---------------------------------------------
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------
# Redirect root to /results
# ---------------------------------------------
@app.get("/")
async def root():
    return RedirectResponse(url="/results")

# ---------------------------------------------
# Serve evaluation results page
# ---------------------------------------------
@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    """
    Serve the evaluation results page with data from evaluate_hybrid.py
    """
    # Run evaluation (e.g., top-k = 3)
    eval_data = evaluate(k=3)

    # Render the Jinja2 template
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "eval_data": eval_data
        }
    )

# ---------------------------------------------
# Optional: Health check endpoint
# ---------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
