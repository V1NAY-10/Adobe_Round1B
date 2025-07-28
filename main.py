# main.py  –  end-to-end Round-1B pipeline
# ----------------------------------------
import os, sys, re, json, pickle, hashlib, mmap
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import Levenshtein
import fitz                        # PyMuPDF

from utils.extractor   import extract_outline
from utils.embedder    import Embedder
from utils.summarizer  import extract_best_snippet
from utils.json_writer import build_output_json, write_json


# ───────────────────────── keyword helpers ──────────────────────────
def extract_keywords(text: str, k: int = 6) -> list[str]:
    words = re.findall(r"[A-Za-z']{3,}", text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:k]


def hybrid_score(q_vec, s_vec, q_tokens, title, prior):
    sem = cosine_similarity([q_vec], [s_vec])[0, 0]
    words = set(re.findall(r"[A-Za-z']{3,}", title.lower()))
    lex  = len(words & q_tokens) / len(words | q_tokens) if words else 0.0
    return 0.45 * sem + 0.45 * lex + 0.10 * prior


# ───────────────────────── de-duplication ───────────────────────────
def dedup_sections(sorted_idx, section_titles, section_embeddings,
                   section_info, threshold_cos=0.90, threshold_lev=5,
                   top_k=5, per_doc_limit=2):
    kept, seen_vecs, seen_titles = [], [], []
    for idx in sorted_idx:
        vec, title = section_embeddings[idx], section_titles[idx].lower()

        if any(cosine_similarity([vec], [v])[0, 0] >= threshold_cos for v in seen_vecs):
            continue
        if any(
            Levenshtein.distance(title, t) <= threshold_lev and
            section_info[idx]["page_number"] == section_info[i]["page_number"]
            for i, t in zip(kept, seen_titles)
        ):
            continue
        doc = section_info[idx]["document"]
        if sum(1 for k in kept if section_info[k]["document"] == doc) >= per_doc_limit:
            continue

        kept.append(idx);  seen_vecs.append(vec);  seen_titles.append(title)
        if len(kept) == top_k:
            break
    return kept


# ───────────────────────── cache utilities ──────────────────────────
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)

def md5_of_file(path: str, block_size: int = 1 << 20) -> str:
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return hashlib.md5(mm).hexdigest()[:16]

def load_pdf_cache(pdf_path: str) -> Tuple[Dict[str, np.ndarray], Path]:
    key   = md5_of_file(pdf_path)
    cache = CACHE_DIR / f"{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes()), cache
    return {}, cache


# ─────────────────────────── main entry ─────────────────────────────
def main(input_json_path: str):
    with open(input_json_path, encoding="utf-8") as f:
        inp = json.load(f)

    persona = inp["persona"]["role"]
    task    = inp["job_to_be_done"]["task"]

    keywords       = extract_keywords(task)
    query_sentence = f"{persona}: {task}. {' '.join(keywords)}"

    embedder   = Embedder()
    query_vec  = embedder.embed(query_sentence)
    query_tok  = set(keywords)
    dim_target = query_vec.shape[0]          # 768 for MPNet

    # ── collect candidate headings ───────────────────────────
    section_titles, section_info = [], []
    for doc_meta in inp["documents"]:
        pdf  = doc_meta["filename"]
        path = os.path.join("input", pdf)
        outline = extract_outline(path)

        for sec in outline["outline"]:
            title = sec["text"].strip()
            if (len(title.split()) < 2 or title[0].islower()) and sec.get("level") != "H1":
                continue
            if len(title) > 80 or title.endswith("."):
                continue
            section_titles.append(title)
            section_info.append({
                "document": pdf,
                "section_title": title,
                "page_number": sec["page"],
                "level": sec.get("level", "")
            })

    # ── graceful fallback if extractor empty ─────────────────
    if not section_titles:
        for doc in inp["documents"]:
            pdf_path = os.path.join("input", doc["filename"])
            try:
                doc_fitz = fitz.open(pdf_path)
            except Exception:
                continue
            for pg in range(min(3, doc_fitz.page_count)):
                first = doc_fitz.load_page(pg).get_text("text").splitlines()
                if not first:
                    continue
                title = first[0].strip()
                if len(title.split()) < 2 or len(title) > 80:
                    continue
                section_titles.append(title)
                section_info.append({
                    "document": doc["filename"],
                    "section_title": title,
                    "page_number": pg + 1,
                    "level": "H1"
                })

    if not section_titles:
        print("No headings found.");  return

    # ── per-PDF embedding cache with dim-check ───────────────
    embeddings = [None] * len(section_titles)
    missing_titles, missing_pos = [], []
    pdf_caches: Dict[str, Dict[str, np.ndarray]] = {}

    for idx, meta in enumerate(section_info):
        pdf_path = os.path.join("input", meta["document"])
        if pdf_path not in pdf_caches:
            pdf_caches[pdf_path], _ = load_pdf_cache(pdf_path)
        cache_dict = pdf_caches[pdf_path]

        title = meta["section_title"]
        emb   = cache_dict.get(title)
        if emb is not None and emb.shape[0] == dim_target:
            embeddings[idx] = emb
        else:
            missing_titles.append(title)
            missing_pos.append(idx)

    if missing_titles:
        new_embs = embedder.embed_batch(missing_titles)
        for pos, emb in zip(missing_pos, new_embs):
            embeddings[pos] = emb
            pdf_path = os.path.join("input", section_info[pos]["document"])
            pdf_caches[pdf_path][section_info[pos]["section_title"]] = emb

    # save updated caches
    for pdf_path, cache_dict in pdf_caches.items():
        _, cache_file = load_pdf_cache(pdf_path)
        cache_file.write_bytes(pickle.dumps(cache_dict))

    section_embeddings = np.vstack(embeddings)

    # ── relevance scoring & filtering ────────────────────────
    scores = []
    for idx, (title, vec, meta) in enumerate(zip(section_titles, section_embeddings, section_info)):
        if cosine_similarity([vec], [query_vec])[0, 0] < 0.05:
            continue
        prior = 1.0 if meta["page_number"] == 1 and meta["level"] == "H1" else 0.0
        scores.append((idx, hybrid_score(query_vec, vec, query_tok, title, prior)))

    if not scores:
        print("No headings survived filtering.");  return

    ranked  = [idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
    top_idx = dedup_sections(ranked, section_titles, section_embeddings, section_info)

    # ── snippet extraction ───────────────────────────────────
    refined_texts = [""] * len(section_info)
    for idx in top_idx:
        sec = section_info[idx]
        pdf_path = os.path.join("input", sec["document"])
        refined_texts[idx] = extract_best_snippet(
            pdf_path, sec["page_number"], query_vec, embedder, keywords
        )

    # ── write JSON output ────────────────────────────────────
    out_json = build_output_json(inp, top_idx, section_info, refined_texts)
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "challenge1b_output.json")
    write_json(out_json, out_path)
    print(f"✅  Output written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py data/your_input.json");  sys.exit(1)
    main(sys.argv[1])
