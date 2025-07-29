# Adobe Hackathon 2025 – Round 1B Submission

## 🧠 Project Title: Offline Heading Extraction & Contextual Matching Pipeline

## 📌 Challenge ID: `round_1b_001`

This project implements an **offline, CPU-only semantic extraction system** to identify the most relevant sections from a bundle of PDF documents based on a given `persona` and `job_to_be_done`.

---

## ✅ Features

- **Zero-internet dependency**: Fully offline inference using `sentence-transformers` and ONNX execution.
- **Lightweight MPNet model**: <420MB total model size.
- **Semantic prompt understanding**: Embeds the persona-task context into a dense vector for matching.
- **Heuristic heading extractor**: Model-free PDF heading extractor based on font size, boldness, structure, etc.
- **Dynamic lexical filtering**: Cosine-based relevance and token overlap used for hybrid scoring.
- **Deduplication**: Avoids duplicate or near-duplicate results across PDFs.
- **Efficient reranking (optional)**: Easy to upgrade with a cross-encoder reranker (MiniLM), but not required in this submission.
- **Robust output JSON**: Output format matches Adobe's specification.

---

## 🗃️ Folder Structure

.
├── data/ # Input JSON(s)
├── input/ # PDF documents
├── output/ # Final output JSON (challenge1b_output.json)
├── models/ # Contains MPNet ONNX model (auto-downloaded)
├── .cache/ # Local PDF embedding caches
├── utils/
│ ├── extractor.py # Heuristic heading extractor
│ ├── embedder.py # MPNet-based semantic embedder
│ ├── summarizer.py # Task-aware snippet extractor
│ ├── json_writer.py # Output builder
│ └── reranker.py # (Optional) MiniLM reranker - not used in this submission
├── main.py # 🔥 Main pipeline
└── README.md # This file

yaml
Copy
Edit

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
Place files

Input PDFs → input/

Input JSON → data/challenge1b_input.json

## 🐳 Docker Instructions

### 📦 Build the Docker Image

```bash
docker build --platform linux/amd64 -t persona-extractor:v1 .
```

### 🚀 Run the Container

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-extractor:v1
```

---

## 🔍 Expected Input

- A folder of PDFs (`input/`)
- A `persona` string + `job-to-be-done` string (read from JSON or hardcoded, depending on design)
- Offline MiniLM & MPNet models located in `models/` subfolders

---

## 📤 Output Format

```json
{
  "metadata": {
    "documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job": "Prepare a literature review on GNNs for drug discovery",
    "timestamp": "2025-07-29T23:58:00"
  },
  "sections": [
    {
      "document": "doc1.pdf",
      "page": 2,
      "section_title": "Methodology Overview",
      "importance_rank": 1
    }
  ],
  "subsections": [
    {
      "document": "doc1.pdf",
      "page": 3,
      "refined_text": "The GNN framework relies on graph convolution layers...",
      "importance_rank": 1
    }
  ]
}
```

---

## 📁 Model Notes

- All transformer models are preloaded into `models/`:
  - `models/minilm/`
  - `models/mpnet/`
- No external downloads or API requests are made inside the container.
- Fully compliant with offline and size constraints.

---

## 🧠 How It Works

1. **Persona + job strings** are converted to embedding vectors.
2. Each section title from each PDF is compared against these vectors.
3. Cosine similarity is used to rank sections by relevance.
4. Subsections under top-ranked sections are summarized and scored.
5. JSON output is created per the challenge specification.

---

## 📜 Multilingual Support ✅

We use `paraphrase-multilingual-MiniLM-L12-v2` to handle non-English documents. This enhances semantic matching for global users.

---

## ❗ Constraints Handled

| Constraint        | Status     |
|-------------------|------------|
| ⏱ ≤ 60 sec        | ✅ Optimized |
| 📦 Model ≤ 1GB     | ✅ (~200MB x 2 models) |
| 🖥 CPU-only        | ✅ Fully CPU compatible |
| 🌐 No Internet     | ✅ Offline execution |
| 🔤 Multilingual     | ✅ Supported |

---
