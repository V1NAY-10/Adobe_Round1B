round1b/
├── input/                          ← Place your PDFs here
├── output/                         ← Output JSONs go here
├── data/                           ← Input JSON (e.g., challenge1b_input.json)
│   └── challenge1b_input.json
├── models/                         ← For storing sentence-transformers locally (optional)
├── utils/                          ← All functional code lives here
│   ├── extractor.py                ← 📄 Extract headings (Round 1A module)
│   ├── embedder.py                 ← 🔢 Convert text → embeddings
│   ├── ranker.py                   ← 🧠 Compare similarity & rank sections
│   ├── summarizer.py               ← ✂️ Extract and clean page content
│   ├── json_writer.py              ← 🧾 Build output JSON
├── main.py                         ← 🔁 Orchestrator (calls all modules)
├── requirements.txt                ← 📦 All pip dependencies
└── README.md                       ← ℹ️ Approach + instructions