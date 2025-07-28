# utils/reranker.py
from pathlib import Path
from typing import List
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

_MODEL, _TOK = None, None
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "reranker"

def _load():
    global _MODEL, _TOK
    if _MODEL is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL_DIR)
        _MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()

def score_pairs(query: str, titles: List[str], batch: int = 8) -> np.ndarray:
    _load()
    scores: List[float] = []
    for i in range(0, len(titles), batch):
        toks = _TOK(
            [query] * len(titles[i : i + batch]),
            titles[i : i + batch],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = _MODEL(**toks).logits.squeeze(1)
        scores.extend(logits.cpu().numpy().tolist())
    return np.array(scores)

def rerank_headings(query: str, candidates: List[str]) -> List[int]:
    scores = score_pairs(query, candidates)
    return list(np.argsort(scores)[::-1])
