"""
app.py — AI Chat with PDF
Upload any PDF → Ask questions → Get answers with page references
100% local using sentence-transformers + FAISS + Gemini (free API)
"""
  
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import numpy as np
import pickle
import hashlib
import os
import re
from sentence_transformers import SentenceTransformer
import faiss
  
# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #08090f; }

  .hero {
    background: linear-gradient(135deg, #0a0d1f 0%, #08090f 60%, #0f0a1a 100%);
    border: 1px solid #1e2040;
    border-radius: 16px;
    padding: 36px 40px;
    text-align: center;
    margin-bottom: 24px;
  }
  .hero h1 { font-size: 40px; font-weight: 700; color: #fff; margin: 0 0 8px; }
  .hero p  { color: #64748b; font-size: 15px; margin: 0; }

  /* Chat bubbles */
  .bubble-user {
    background: linear-gradient(135deg, #1e40af, #2563eb);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 14px;
    line-height: 1.6;
  }
  .bubble-ai {
    background: #0f1020;
    border: 1px solid #1e2040;
    color: #d4d8f0;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px;
    margin: 8px 60px 8px 0;
    font-size: 14px;
    line-height: 1.7;
  }
  .bubble-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .source-chip {
    display: inline-block;
    background: #1e2040;
    border: 1px solid #2e3060;
    color: #818cf8;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    margin: 4px 3px 0 0;
    font-weight: 600;
  }

  .stat-pill {
    display: inline-block;
    background: #0f1020;
    border: 1px solid #1e2040;
    color: #818cf8;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 3px;
  }

  .suggested-q {
    background: #0f1020;
    border: 1px solid #1e2040;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    color: #94a3b8;
    cursor: pointer;
    margin: 5px 0;
    transition: border-color 0.2s;
  }
  .suggested-q:hover { border-color: #4f46e5; color: #c7d2fe; }

  div.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 12px 24px;
    font-size: 14px;
    width: 100%;
  }
  div.stButton > button:hover { opacity: 0.85; }

  .stTextInput input {
    background: #0f1020 !important;
    border-color: #1e2040 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
  }
  .processing-box {
    background: #0a0f1e;
    border: 1px solid #1e3060;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
    font-size: 13px;
    color: #64748b;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400    # words per chunk
CHUNK_OVERLAP = 80     # word overlap between chunks
TOP_K         = 5      # number of chunks to retrieve per question
CACHE_DIR     = ".pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Load embedding model (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ── PDF processing ─────────────────────────────────────────────────────────────
def pdf_to_text_pages(pdf_bytes: bytes) -> list[dict]:
    """Extract text from each page of a PDF."""
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages  = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages

def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split pages into overlapping word-level chunks."""
    chunks = []
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end        = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            if len(chunk_text.strip()) > 50:
                chunks.append({"page": p["page"], "text": chunk_text})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    """Embed all chunks and build a FAISS inner-product index."""
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index, embeddings

def retrieve(question: str, chunks, index, top_k=TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a question."""
    q_emb = embedder.encode([question], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            results.append({**chunks[idx], "score": float(score)})
    return results

# ── PDF caching ────────────────────────────────────────────────────────────────
def get_pdf_hash(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()

def load_cache(pdf_hash: str):
    path = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def save_cache(pdf_hash: str, data: dict):
    path = os.path.join(CACHE_DIR, f"{pdf_hash}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)

# ── Gemini answer generation ───────────────────────────────────────────────────
def ask_gemini(question: str, context_chunks: list[dict], api_key: str, chat_history: list) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={"temperature": 0.3, "max_output_tokens": 1024},
    )

    context = "\n\n".join([
        f"[Page {c['page']}]\n{c['text']}" for c in context_chunks
    ])

    history_text = ""
    if chat_history:
        last_turns = chat_history[-4:]  # last 2 Q&A pairs
        history_text = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in last_turns
        ])

    prompt = f"""You are a helpful AI assistant that answers questions about a PDF document.
Answer ONLY based on the context provided below. If the answer is not in the context, say "I couldn't find that in the document."
Be concise, accurate, and cite page numbers when relevant.

DOCUMENT CONTEXT:
{context}

{"RECENT CONVERSATION:" + history_text if history_text else ""}

USER QUESTION: {question}

Answer:"""

    response = model.generate_content(prompt)
    return response.text.strip()

# ── Auto-generate suggested questions ─────────────────────────────────────────
def generate_suggestions(chunks: list[dict], api_key: str) -> list[str]:
    try:
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        sample = " ".join([c["text"] for c in chunks[:5]])[:2000]
        resp   = model.generate_content(
            f"Based on this document excerpt, generate 4 short, specific questions a reader might ask. "
            f"Return only the questions as a numbered list, nothing else.\n\n{sample}"
        )
        lines = [l.strip() for l in resp.text.strip().split("\n") if l.strip()]
        return [re.sub(r"^\d+[\.\)]\s*", "", l) for l in lines if len(l) > 10][:4]
    except Exception:
        return [
            "What is the main topic of this document?",
            "What are the key findings or conclusions?",
            "Can you summarise the most important points?",
            "What recommendations are made in this document?",
        ]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 AI Chat with PDF")
    st.markdown("---")

    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input(
        "Free Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get FREE at https://aistudio.google.com"
    )
    if not api_key:
        st.info("🆓 Get a **free** key at [aistudio.google.com](https://aistudio.google.com) — no credit card!")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    top_k_setting = st.slider("Chunks to retrieve per question", 3, 8, TOP_K)
    show_sources  = st.checkbox("Show source chunks", value=True)

    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
- Ask **specific questions** for better answers
- Include **page numbers** in your question for targeted lookup
- Ask for **summaries** of specific sections
- Works best with text-based PDFs (not scanned images)
    """)

    st.markdown("---")
    st.markdown("### 🔒 Privacy")
    st.markdown("PDF text is processed **locally**. Only your question + relevant excerpts are sent to Gemini.")

    if st.button("🗑️ Clear Chat History"):
        st.session_state["messages"] = []
        st.rerun()

# ── Main UI ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📄 AI Chat with PDF</h1>
  <p>Upload any PDF → Ask questions in plain English → Get answers with page references</p>
</div>
""", unsafe_allow_html=True)

# ── File upload ────────────────────────────────────────────────────────────────
uploaded_pdf = st.file_uploader(
    "📂 Upload a PDF file",
    type=["pdf"],
    help="Works best with text-based PDFs. Scanned image PDFs need OCR preprocessing."
)

# ── Process PDF ────────────────────────────────────────────────────────────────
if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()
    pdf_hash  = get_pdf_hash(pdf_bytes)

    # Check cache
    cached = load_cache(pdf_hash)
    if cached:
        chunks    = cached["chunks"]
        index     = cached["index"]
        pdf_info  = cached["info"]
        st.success(f"✅ **{uploaded_pdf.name}** loaded from cache")
    else:
        with st.spinner("📖 Reading and indexing your PDF..."):
            pages    = pdf_to_text_pages(pdf_bytes)
            chunks   = chunk_pages(pages)
            index, _ = build_index(chunks)
            pdf_info = {
                "name":   uploaded_pdf.name,
                "pages":  len(pages),
                "chunks": len(chunks),
                "words":  sum(len(p["text"].split()) for p in pages),
            }
            save_cache(pdf_hash, {"chunks": chunks, "index": index, "info": pdf_info})

        st.success(f"✅ **{uploaded_pdf.name}** indexed successfully!")

    # Stats row
    st.markdown(
        f'<div style="margin:12px 0 20px;">'
        f'<span class="stat-pill">📄 {pdf_info["pages"]} pages</span>'
        f'<span class="stat-pill">🧩 {pdf_info["chunks"]} chunks</span>'
        f'<span class="stat-pill">📝 {pdf_info["words"]:,} words</span>'
        f'<span class="stat-pill">🔍 Semantic search ready</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Suggested questions
    if api_key and "suggestions" not in st.session_state:
        with st.spinner("💡 Generating suggested questions..."):
            st.session_state["suggestions"] = generate_suggestions(chunks, api_key)

    if "suggestions" in st.session_state:
        st.markdown("**💡 Suggested questions — click to ask:**")
        s_cols = st.columns(2)
        for i, suggestion in enumerate(st.session_state.get("suggestions", [])):
            with s_cols[i % 2]:
                if st.button(f"❓ {suggestion}", key=f"sug_{i}"):
                    st.session_state["pending_question"] = suggestion

    st.markdown("---")

    # ── Chat interface ─────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-label" style="color:#6366f1;margin-left:4px;">YOU</div>'
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            pages_cited = msg.get("pages", [])
            chips = "".join([f'<span class="source-chip">📄 Page {p}</span>' for p in sorted(set(pages_cited))])
            st.markdown(
                f'<div class="bubble-label" style="color:#818cf8;margin-left:4px;">AI</div>'
                f'<div class="bubble-ai">{msg["content"]}'
                + (f'<div style="margin-top:10px;">{chips}</div>' if chips else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

            # Show source chunks if enabled
            if show_sources and "sources" in msg:
                with st.expander(f"📚 View {len(msg['sources'])} source chunks used"):
                    for src in msg["sources"]:
                        st.markdown(f"**Page {src['page']}** *(relevance: {src['score']:.2f})*")
                        st.markdown(f"> {src['text'][:300]}...")
                        st.markdown("---")

    # ── Question input ─────────────────────────────────────────────────────────
    pending = st.session_state.pop("pending_question", None)

    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Ask anything about your PDF",
                value=pending or "",
                placeholder="e.g. What are the main conclusions? / Summarise page 3 / What does X mean?",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤")

    if submitted and user_input.strip():
        if not api_key:
            st.error("⚠️ Please add your free Gemini API key in the sidebar.")
        else:
            question = user_input.strip()

            # Add user message
            st.session_state["messages"].append({"role": "user", "content": question})

            with st.spinner("🔍 Searching document + generating answer..."):
                try:
                    # Retrieve relevant chunks
                    relevant = retrieve(question, chunks, index, top_k=top_k_setting)

                    # Generate answer
                    answer = ask_gemini(
                        question,
                        relevant,
                        api_key,
                        st.session_state["messages"][:-1],
                    )

                    pages_cited = [c["page"] for c in relevant]

                    # Add AI message
                    st.session_state["messages"].append({
                        "role":    "assistant",
                        "content": answer,
                        "pages":   pages_cited,
                        "sources": relevant,
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")

            st.rerun()

else:
    # Empty state
    st.markdown("""
<div style="text-align:center;padding:60px 20px;">
  <div style="font-size:72px;margin-bottom:16px;">📄</div>
  <h3 style="color:#475569;">Upload a PDF to get started</h3>
  <p style="color:#334155;font-size:14px;max-width:500px;margin:0 auto;">
    Works with research papers, books, contracts, reports, manuals — any text-based PDF.
    Ask questions in plain English and get answers with exact page references.
  </p>
</div>

<div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;margin-top:32px;">
""", unsafe_allow_html=True)

    for example in ["📚 Research paper", "📑 Legal contract", "📘 Textbook chapter", "📊 Annual report", "📋 User manual"]:
        st.markdown(f'<span class="stat-pill">{example}</span>', unsafe_allow_html=True)
