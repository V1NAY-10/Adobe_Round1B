"""
summarizer.py
=============

Given a PDF path and a candidate page number, return the most relevant
paragraph (≤250 words) to the query vector.
"""

import re, fitz
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _split_into_paragraphs(page_text: str) -> List[str]:
    blocks = re.split(r"\n\s*\n", page_text.strip())
    # remove super-short or very long blocks
    return [b.strip().replace("\n", " ") for b in blocks if 20 < len(b) < 2000]


def _trim(words: List[str], max_len: int = 250) -> str:
    """Ensure snippet never exceeds max_len words."""
    if len(words) <= max_len:
        return " ".join(words)
    return " ".join(words[:max_len]) + " [...]"


# ─────────────────────────────────────────────────────────────
def extract_best_snippet(
    pdf_path: str,
    page_number: int,
    query_vec: np.ndarray,
    embedder,
    keywords: list[str]
) -> str:
    """
    Look at page_number-1, page_number, page_number+1 and return
    the paragraph with highest cosine similarity to the query.
    """

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    candidates = []
    for p in (page_number - 2, page_number - 1, page_number, page_number + 1):
        if p < 0 or p >= doc.page_count:
            continue
        text = doc.load_page(p).get_text("text")
        for para in _split_into_paragraphs(text):
            candidates.append(para)

    if not candidates:
        return ""

    # quick keyword boost: keep only paragraphs that mention ≥1 keyword
    if keywords:
        kw_pat = re.compile("|".join(map(re.escape, keywords)), re.I)
        filtered = [c for c in candidates if kw_pat.search(c)]
        if filtered:
            candidates = filtered

    embs = embedder.embed_batch(candidates)
    sims = cosine_similarity([query_vec], embs)[0]
    best_idx = int(np.argmax(sims))
    best_p   = candidates[best_idx]

    return _trim(best_p.split(), max_len=250)
