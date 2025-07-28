# utils/embedder.py

from sentence_transformers import SentenceTransformer

# instead of a remote name like "sentence-transformers/all-mpnet-base-v2"
# use the local model path:
_MODEL_NAME = "models/mpnet"   # ✅ <- points to your local files

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(_MODEL_NAME)

    def embed(self, text: str):
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: list[str]):
        return self.model.encode(texts, convert_to_numpy=True, batch_size=16, show_progress_bar=False)
