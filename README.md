# 📄 AI Chat with PDF

<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=200&section=header&text=AI%20Chat%20with%20PDF&fontSize=50&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Ask%20Any%20Question%20About%20Any%20PDF%20%7C%20RAG%20%2B%20Semantic%20Search%20%2B%20Gemini%20AI&descAlignY=55&descSize=15)

<p>
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/RAG-Architecture-6366F1?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00ADD8?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Gemini%201.5%20Flash-Free%20API-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge"/>
</p>

<p>
  <b>Upload any PDF → Ask questions in plain English → Get accurate answers with exact page references.</b><br/>
  Built on a full RAG (Retrieval-Augmented Generation) pipeline using FAISS vector search + Gemini AI.
</p>

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [How It Works](#-how-it-works) • [Architecture](#-architecture) • [FAQ](#-faq)

</div>

---

## 🌟 Why This Project?

Tired of scrolling through 100-page PDFs to find one answer? This tool lets you **have a conversation with any document**:

- 📚 Research papers — "What methodology did they use?"
- 📑 Legal contracts — "What are the termination clauses?"
- 📘 Textbooks — "Explain the concept on page 45 in simple terms"
- 📊 Annual reports — "What was the revenue growth in Q3?"
- 📋 User manuals — "How do I configure the network settings?"

Built using **RAG (Retrieval-Augmented Generation)** — the same architecture used by enterprise document AI tools like ChatPDF, Adobe AI Assistant, and Notion AI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic Search** | Finds relevant sections by *meaning*, not just keywords |
| 📄 **Page References** | Every answer includes the exact page numbers it came from |
| 💬 **Multi-turn Chat** | Follow-up questions maintain conversation context |
| 💡 **Auto Suggestions** | AI generates 4 relevant questions based on your document |
| ⚡ **PDF Caching** | Indexed PDFs are cached — re-upload is instant |
| 📚 **Source Viewer** | Expand to see the exact chunks used to generate each answer |
| 🧩 **Chunking Engine** | Overlapping word-level chunks preserve context across boundaries |
| 🔒 **Privacy First** | PDF processed locally; only question + excerpts sent to Gemini |
| 🗑️ **Clear Chat** | Reset conversation without re-processing the PDF |

---

## 🖥️ Demo

```
╔══════════════════════════════════════════════════════════════════╗
║  📄 AI Chat with PDF                                             ║
║  research_paper.pdf │ 📄 32 pages │ 🧩 89 chunks │ 📝 12,450 words ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  💡 Suggested questions:                                         ║
║  ❓ What methodology was used in this study?                     ║
║  ❓ What are the main findings?                                  ║
║                                                                  ║
║  ──────────────────────────────────────────────────────────────  ║
║                                                                  ║
║  YOU                                                             ║
║  ╭─────────────────────────────────────────────────────────╮    ║
║  │ What are the key limitations of this research?          │    ║
║  ╰─────────────────────────────────────────────────────────╯    ║
║                                                                  ║
║  AI                                                              ║
║  ╭─────────────────────────────────────────────────────────╮    ║
║  │ The study identifies three key limitations:             │    ║
║  │ 1. Small sample size of only 42 participants...         │    ║
║  │ 2. Limited to English-language sources only...          │    ║
║  │ 3. Cross-sectional design prevents causal claims...     │    ║
║  │                                                         │    ║
║  │ 📄 Page 28  📄 Page 29                                  │    ║
║  ╰─────────────────────────────────────────────────────────╯    ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📦 Installation

### Prerequisites
- Python 3.9+ → [Download](https://www.python.org/downloads/)
- Free Gemini API key → [Get here](https://aistudio.google.com) *(no credit card)*

### Step 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/ai-chat-with-pdf.git
cd ai-chat-with-pdf
```

### Step 2 — Virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **First run:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~90MB). Internet required once — then fully offline.

### Step 4 — Run
```bash
streamlit run app.py
```

Opens at **http://localhost:8501** 🎉

---

## 🧠 How It Works

This project implements **RAG (Retrieval-Augmented Generation)** — the industry-standard architecture for document Q&A:

### The RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  INDEXING (done once per PDF)                                   │
│                                                                 │
│  PDF Upload                                                     │
│      │                                                          │
│      ▼                                                          │
│  PyMuPDF extracts text page by page                            │
│      │                                                          │
│      ▼                                                          │
│  Text split into overlapping chunks (400 words, 80 overlap)    │
│      │                                                          │
│      ▼                                                          │
│  sentence-transformers encodes each chunk → 384-dim vectors    │
│      │                                                          │
│      ▼                                                          │
│  FAISS IndexFlatIP stores all vectors                          │
│      │                                                          │
│      ▼                                                          │
│  Cached to disk (.pdf_cache/) for instant re-use               │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  RETRIEVAL + GENERATION (every question)                        │
│                                                                 │
│  User types question                                            │
│      │                                                          │
│      ▼                                                          │
│  Encode question → 384-dim vector                              │
│      │                                                          │
│      ▼                                                          │
│  FAISS inner-product search → top 5 most relevant chunks       │
│      │                                                          │
│      ▼                                                          │
│  Build prompt: context chunks + chat history + question        │
│      │                                                          │
│      ▼                                                          │
│  Gemini 1.5 Flash generates answer                             │
│      │                                                          │
│      ▼                                                          │
│  Display answer + page references + source chunks              │
└─────────────────────────────────────────────────────────────────┘
```

### Why these specific choices?

| Component | Choice | Why |
|---|---|---|
| **PDF Parser** | PyMuPDF | Fastest, handles complex layouts |
| **Embeddings** | all-MiniLM-L6-v2 | Best speed/accuracy for free, 384 dims |
| **Vector DB** | FAISS (CPU) | Zero setup, runs locally, blazing fast |
| **LLM** | Gemini 1.5 Flash | Free tier, 1M token context, accurate |
| **Chunking** | 400 words + 80 overlap | Balances context preservation & precision |
| **Similarity** | Inner Product (normalised) | Equivalent to cosine similarity, faster |

---

## 📁 Project Structure

```
ai-chat-with-pdf/
│
├── app.py              # 🧠 Main Streamlit app — full RAG pipeline
├── requirements.txt    # 📦 6 dependencies
├── .gitignore          # 🚫 Excludes cache, model downloads
├── LICENSE             # 📄 MIT License
└── README.md           # 📖 You are here
│
└── .pdf_cache/         # ⚡ Auto-created — cached FAISS indexes
    └── <md5hash>.pkl
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io) | 1.35 | Web UI |
| [PyMuPDF](https://pymupdf.readthedocs.io) | 1.24 | PDF text extraction |
| [sentence-transformers](https://sbert.net) | 3.0 | Local text embeddings (all-MiniLM-L6-v2) |
| [FAISS](https://github.com/facebookresearch/faiss) | 1.8 | Vector similarity search (Facebook AI) |
| [Google Gemini](https://aistudio.google.com) | 1.5 Flash | Answer generation |
| [NumPy](https://numpy.org) | 1.26 | Vector operations |

---

## 🤔 FAQ

**Q: Does it work on scanned PDFs?**
> Scanned PDFs (image-based) won't work without OCR preprocessing. Text-based PDFs (most research papers, reports, contracts) work perfectly.

**Q: What's the maximum PDF size?**
> No hard limit — tested up to 500 pages. Larger PDFs take longer to index on first upload but are instant after caching.

**Q: Is my PDF data private?**
> Your PDF is processed locally by PyMuPDF. Only the question + the 5 most relevant text excerpts (not the whole PDF) are sent to Gemini. Raw PDF bytes never leave your machine.

**Q: Can I run it fully offline?**
> After the first run (which downloads the embedding model), the indexing and retrieval work completely offline. Only the final Gemini call needs internet.

**Q: Why does it sometimes give wrong answers?**
> RAG accuracy depends on chunk quality. If an answer spans multiple chunks or requires reasoning across distant sections, it may miss nuance. Try rephrasing your question or asking for a specific page.

**Q: Can I swap Gemini for a local LLM?**
> Yes! Replace the `ask_gemini()` function with an Ollama call (`llama3`, `mistral`, etc.) for a fully offline experience.

---

## 🗺️ Roadmap

- [ ] 🌐 URL input — paste a webpage URL instead of uploading a PDF
- [ ] 📊 Multiple PDF support — chat across several documents at once
- [ ] 🔍 Keyword highlight — highlight the source text directly in a PDF viewer
- [ ] 🦙 Ollama integration — fully offline mode with local LLMs
- [ ] 📤 Export chat as PDF transcript
- [ ] 🌍 Multi-language support
- [ ] 🎙️ Voice input for questions

---

## 🤝 Contributing

1. Fork this repo
2. Create a branch: `git checkout -b feature/your-idea`
3. Commit: `git commit -m 'feat: your feature'`
4. Push & open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — inspired the RAG architecture pattern
- [sentence-transformers](https://sbert.net) — by UKP Lab, TU Darmstadt
- [FAISS](https://github.com/facebookresearch/faiss) — by Facebook AI Research
- [PyMuPDF](https://pymupdf.readthedocs.io) — for rock-solid PDF parsing

---

<div align="center">

**⭐ If this saved you time, star the repo — it really helps!**

Made with ❤️ and Python

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,12,20&height=100&section=footer)

</div>
