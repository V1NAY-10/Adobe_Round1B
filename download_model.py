"""
download_model.py ───────────────────────────────────────────
Fetch the Tier-L encoder (multi-qa-mpnet-base-dot-v1) and save
it into models/encoder  so the runtime stays 100 % offline.
Run this once before you build / run the Docker image.
"""

from pathlib import Path
import shutil
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEST       = Path("models") / "encoder"

def main() -> None:
    print(f"⏬  Downloading {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)

    if DEST.exists():
        shutil.rmtree(DEST)
    model.save(str(DEST))
    print(f"✅  Saved to {DEST}  (size ≈ 438 MB)")

if __name__ == "__main__":
    main()
