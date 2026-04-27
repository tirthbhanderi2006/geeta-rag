"""
MODULE 3: RAG Chat Interface (Streamlit)
- Conversational chat UI with message history
- RAG retrieval panel showing source passages
- Multilingual support (English, Gujarati, Sanskrit, Hindi)
- Groq LLM backend (LLaMA 3.3 70B)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project modules
from ingest import ingest_pdf
from utils import build_vector_store, retrieve_chunks, keyword_search, get_chroma_collection


# ---------------------------------------------------------------------------
# Groq LLM backend
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, context: str) -> str:
    """Build the RAG prompt with retrieved context — multilingual aware."""
    prompt = (
        "You are an expert multilingual assistant specialized in spiritual and "
        "philosophical texts, particularly the Bhagavad Gita (Yatharth Geeta).\n"
        "You understand English, Gujarati, Sanskrit, and Hindi.\n\n"
        "Instructions:\n"
        "- Answer the question using ONLY the provided context passages below.\n"
        "- The context is extracted via OCR from a Gujarati Bhagavad Gita commentary PDF.\n"
        "- Some text may contain OCR artifacts or mixed scripts. Do your best to interpret the meaning.\n"
        "- Be precise, detailed, and cite page numbers when possible.\n"
        "- Use markdown formatting for well-structured answers.\n\n"
        "CRITICAL LANGUAGE RULE:\n"
        "- If the user asks in ENGLISH, answer ONLY in ENGLISH.\n"
        "- If the user asks in GUJARATI, answer ONLY in GUJARATI script.\n"
        "- If the user asks in HINDI, answer ONLY in HINDI (Devanagari script).\n"
        "- If the user asks in SANSKRIT, answer in SANSKRIT or Hindi.\n"
        "- ALWAYS match the user's language. Never mix languages.\n"
        "- If the answer is not in the context, say you do not have enough information.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (in the SAME language as the question):"
    )
    return prompt


def answer_with_groq(question: str, context: str, chat_history: list = None) -> str:
    """Generate answer using Groq API with LLaMA 3.3 70B."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ GROQ_API_KEY is not set in your .env file. Please add your Groq API key."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = build_rag_prompt(question, context)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual scholarly assistant for the Bhagavad Gita (Yatharth Geeta). "
                    "You can read and respond in English, Gujarati, Sanskrit, and Hindi. "
                    "The source document is OCR-extracted from a Gujarati PDF and may contain some OCR noise. "
                    "CRITICAL: Always answer in the SAME language the user asks in. "
                    "English question = English answer. "
                    "Gujarati question = Gujarati answer in Gujarati script. "
                    "Hindi question = Hindi answer in Devanagari script. "
                    "Use markdown formatting for clear, well-structured answers."
                ),
            },
        ]

        # Add last 6 turns of conversation for context continuity
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API error: {e}", exc_info=True)
        return f"❌ Groq API error: {e}"


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def rag_retrieve(question: str, top_k: int, search_mode: str, pages: list) -> Tuple[List[Dict], str]:
    """Retrieve relevant chunks from the vector store."""
    try:
        if search_mode == "Semantic":
            chunks = retrieve_chunks(question, top_k=top_k, persist_dir="chroma_db")
        elif search_mode == "Keyword":
            chunks = keyword_search(question, pages, max_results=top_k)
        else:
            # Hybrid
            sem_chunks = retrieve_chunks(question, top_k=top_k // 2 + 1, persist_dir="chroma_db")
            kw_chunks = keyword_search(question, pages, max_results=top_k // 2 + 1)
            seen = set()
            chunks = []
            for c in sem_chunks + kw_chunks:
                key = c["text"][:100]
                if key not in seen:
                    seen.add(key)
                    chunks.append(c)
            chunks = chunks[:top_k]

        context = "\n\n---\n\n".join([c["text"] for c in chunks])
        return chunks, context
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        return [], ""


def load_existing_data():
    """Try to load pre-ingested data on startup."""
    json_files = list(Path("data").glob("*_pages.json"))
    if not json_files:
        return None, None, 0

    latest = sorted(json_files)[-1]
    try:
        with open(latest, "r", encoding="utf-8") as f:
            pages = json.load(f)

        collection = get_chroma_collection("chroma_db")
        n = collection.count()
        source_name = latest.stem.replace("_pages", "")
        return pages, source_name, n
    except Exception:
        return None, None, 0


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="📖 Multilingual RAG Chat",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white;
    }

    /* RAG card styling */
    .rag-card {
        background: #f0f4ff;
        border-left: 4px solid #4f46e5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .rag-card-header {
        font-weight: 600;
        color: #4f46e5;
        margin-bottom: 6px;
    }
    .rag-card-text {
        color: #374151;
        line-height: 1.5;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .status-connected {
        background: #d1fae5;
        color: #065f46;
    }
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Header */
    .app-header {
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pages" not in st.session_state:
    st.session_state.pages = []

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "source_name" not in st.session_state:
    st.session_state.source_name = ""

if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0

if "last_rag_chunks" not in st.session_state:
    st.session_state.last_rag_chunks = []

# Auto-load existing data on first run
if not st.session_state.ingested:
    pages, source_name, n_chunks = load_existing_data()
    if pages:
        st.session_state.pages = pages
        st.session_state.ingested = True
        st.session_state.source_name = source_name
        st.session_state.n_chunks = n_chunks


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📖 RAG Chat Assistant")
    st.markdown("*Multilingual Knowledge Exploration*")
    st.divider()

    # Status indicators
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        st.success("🤖 Groq API: Connected")
    else:
        st.error("🤖 Groq API: Missing Key")

    if st.session_state.ingested:
        st.success(f"📄 Document: `{st.session_state.source_name}`")
        st.info(f"📊 Pages: {len(st.session_state.pages)} | Chunks: {st.session_state.n_chunks}")
    else:
        st.warning("📄 No document loaded")

    st.divider()

    # Search settings
    st.markdown("### ⚙️ Search Settings")
    search_mode = st.radio(
        "Search Mode",
        options=["Semantic", "Keyword", "Hybrid"],
        index=0,
        help="Semantic uses AI embeddings, Keyword does exact matching, Hybrid combines both.",
    )
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)

    st.divider()

    # Upload section
    st.markdown("### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Supports scanned & digital PDFs in English, Gujarati, Sanskrit",
    )

    if uploaded_file and st.button("🚀 Ingest & Index", type="primary", use_container_width=True):
        # Save uploaded file to temp location
        temp_path = Path("data") / uploaded_file.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(uploaded_file.getvalue())

        with st.spinner("📄 Extracting text & building index..."):
            try:
                result = ingest_pdf(str(temp_path), output_dir="data")
                n_chunks = build_vector_store(result["pages"], persist_dir="chroma_db")

                st.session_state.pages = result["pages"]
                st.session_state.ingested = True
                st.session_state.source_name = Path(uploaded_file.name).stem
                st.session_state.n_chunks = n_chunks
                st.session_state.messages = []  # Reset chat on new document
                st.session_state.last_rag_chunks = []

                st.success(f"✅ Ingested {result['total_pages']} pages, {n_chunks} chunks indexed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ingestion failed: {e}")

    st.divider()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_rag_chunks = []
        st.rerun()

    st.divider()
    st.markdown(
        """
        <div style='text-align:center; font-size:0.75em; opacity:0.7;'>
        🤖 Groq LLaMA 3.3 70B<br>
        🔍 ChromaDB + Multilingual Embeddings<br>
        🌐 EN · ગુજરાતી · संस्कृत · हिन्दी
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Layout: chat (left) + RAG panel (right)
chat_col, rag_col = st.columns([7, 3])

with chat_col:
    # Header
    st.markdown("""
    <div class="app-header">
        <h2>💬 Chat with your Document</h2>
        <p style="color: #6b7280; margin-top: -10px;">
            Ask in <strong>English</strong>, <strong>ગુજરાતી</strong>, <strong>संस्कृत</strong>, or <strong>हिन्दी</strong> — I'll answer in your language
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your document... (any language)"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.ingested:
            error_msg = "⚠️ No document loaded yet. Please upload a PDF using the sidebar first."
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # RAG retrieval
            with st.spinner("🔍 Searching document..."):
                chunks, context = rag_retrieve(
                    prompt, top_k, search_mode, st.session_state.pages
                )
                st.session_state.last_rag_chunks = chunks

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("🤖 Generating answer..."):
                    if context:
                        answer = answer_with_groq(
                            prompt, context,
                            chat_history=st.session_state.messages[:-1]  # exclude current user msg
                        )
                    else:
                        answer = "I couldn't find any relevant passages in the document. Try rephrasing your question or using a different search mode."

                    # Add source indicator
                    source_note = f"\n\n---\n📚 *Based on {len(chunks)} retrieved passage(s) from `{st.session_state.source_name}` | Search: {search_mode}*"
                    full_answer = answer + source_note
                    st.markdown(full_answer)

            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            st.rerun()


# ---------------------------------------------------------------------------
# RAG Sources Panel (right column)
# ---------------------------------------------------------------------------

with rag_col:
    st.markdown("### 📚 Retrieved Passages")
    st.caption("Source context used by the LLM to generate answers")
    st.divider()

    if st.session_state.last_rag_chunks:
        for i, chunk in enumerate(st.session_state.last_rag_chunks, 1):
            score = chunk.get("score", "N/A")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            text_preview = chunk["text"][:400]
            if len(chunk["text"]) > 400:
                text_preview += "..."

            st.markdown(
                f"""
                <div class="rag-card">
                    <div class="rag-card-header">
                        📄 [{i}] Page {chunk['page_num']} &nbsp;·&nbsp; {chunk['strategy']} &nbsp;·&nbsp; Score: {score_str}
                    </div>
                    <div class="rag-card-text">{text_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(f"Showing {len(st.session_state.last_rag_chunks)} passages | Mode: {search_mode}")
    else:
        st.markdown(
            """
            <div style="text-align: center; padding: 40px 20px; color: #9ca3af;">
                <p style="font-size: 2em; margin-bottom: 10px;">🔍</p>
                <p>Ask a question to see<br>retrieved passages here</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Document stats expander
    if st.session_state.ingested:
        with st.expander("📊 Document Stats"):
            pages = st.session_state.pages
            scanned = sum(1 for p in pages if p.get("scanned"))
            langs = {}
            for p in pages:
                l = p.get("language_detected", "unknown")
                langs[l] = langs.get(l, 0) + 1

            st.markdown(f"**Source**: `{st.session_state.source_name}`")
            col1, col2 = st.columns(2)
            col1.metric("Total Pages", len(pages))
            col2.metric("Chunks Indexed", st.session_state.n_chunks)
            col1.metric("Scanned (OCR)", scanned)
            col2.metric("Digital", len(pages) - scanned)

            st.markdown("**Languages detected:**")
            for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
                st.markdown(f"- `{lang}`: {count} pages")