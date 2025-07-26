# utils/ranker.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_sections(query_embedding: np.ndarray, section_texts: list[str], section_embeddings: np.ndarray, top_k=5):
    """
    Given a query embedding and a list of section embeddings,
    return the indices of the top-k most relevant sections.
    """

    similarities = cosine_similarity([query_embedding], section_embeddings)[0]  # shape: (num_sections,)

    # Get top-k indices (sorted by highest similarity)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Package the results with similarity scores
    ranked = []
    for rank, idx in enumerate(top_indices, start=1):
        ranked.append({
            "index": idx,
            "score": float(similarities[idx]),
            "importance_rank": rank
        })

    return ranked
