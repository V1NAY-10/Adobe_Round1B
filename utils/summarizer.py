import fitz, re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _sentence_split(text):
    return re.split(r'(?<=[\.!?])\s+', text.strip())

def extract_best_snippet(pdf_path, page_num, query_vec, embedder, keywords):
    try:
        doc = fitz.open(pdf_path)
        text = doc.load_page(page_num-1).get_text("text")
    except Exception:
        return ""

    sents = [s for s in _sentence_split(text) if len(s.split())>=6]
    if not sents: return ""

    sent_vecs = embedder.embed_batch(sents)
    sims      = cosine_similarity(sent_vecs, [query_vec]).squeeze(1)
    best_idx  = int(np.argmax(sims))
    best      = sents[best_idx]

    # bold keywords
    for kw in keywords:
        best = re.sub(f"(?i)({re.escape(kw)})", r"**\1**", best, 1)
    return best
