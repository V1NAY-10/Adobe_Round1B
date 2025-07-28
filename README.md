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

Run pipeline

bash
Copy
Edit
python main.py data/challenge1b_input.json
Output

Saved to output/challenge1b_output.json

🔒 Offline Guarantee
All model artifacts (MPNet-ONNX) are automatically cached and reused. No internet access is required after the first setup.

📋 Constraints Satisfied
Requirement	Status
CPU-only	✅
Offline-only (no API calls)	✅
Model size < 1GB	✅ (~420MB)
No hardcoded rule exceptions	✅
Domain-agnostic	✅
Robust to changing inputs	✅

✨ Example Prompt Format
json
Copy
Edit
{
  "persona": { "role": "Chef" },
  "job_to_be_done": { "task": "List only dinner options from the PDFs" },
  "documents": [
    { "filename": "Dinner Ideas - Mains_1.pdf", "title": "Dinner Ideas - Mains_1" },
    ...
  ]
}
