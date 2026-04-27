"""
MODULE 3: QA / Exploration Tool
Gradio Chat UI with:
- File upload + ingestion trigger
- Conversational chat interface with Groq LLM
- RAG context display alongside LLM answers
- Multilingual query support (English, Gujarati, Sanskrit, Hindi)
- Groq LLM backend (LLaMA 3.3 70B)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import gradio as gr

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
    return f"""You are an expert multilingual assistant specialized in spiritual and philosophical texts, particularly the Bhagavad Gita.
You understand English, Gujarati (ગુજરાતી), Sanskrit (संस्कृत), and Hindi (हिन्दी).

Instructions:
- Answer the question using ONLY the provided context passages below.
- If the context contains text in Gujarati or Sanskrit, you may translate or explain it in the language the user asked the question in.
- If the user asks in Gujarati, reply in Gujarati. If in English, reply in English. If in Hindi, reply in Hindi. Match the user's language.
- If the answer is not in the context, say "I don't have enough information in the retrieved passages."
- Be precise, detailed, and cite page numbers when possible.
- Format your answer with clear structure using markdown (headings, bullet points, bold text).

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


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
                    "You are a multilingual scholarly assistant for the Bhagavad Gita. "
                    "You can read and respond in English, Gujarati (ગુજરાતી), Sanskrit (संस्कृत), and Hindi (हिन्दी). "
                    "Always base your answers strictly on the provided context. "
                    "Use markdown formatting for clear, well-structured answers. "
                    "When the user greets you, respond warmly and let them know what you can help with."
                ),
            },
        ]

        # Add conversation history for context continuity (last 6 turns)
        if chat_history:
            for msg in chat_history[-6:]:
                if msg["role"] in ("user", "assistant"):
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
# Core RAG logic
# ---------------------------------------------------------------------------

_session_state = {
    "pages": [],
    "ingested": False,
    "source_name": "",
}


def ingest_and_index(pdf_file, progress=None) -> str:
    """Ingest uploaded PDF and build vector store."""
    if pdf_file is None:
        return "❌ No file uploaded."

    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    source_name = Path(pdf_path).stem

    try:
        logger.info("📄 Extracting text from PDF...")
        result = ingest_pdf(pdf_path, output_dir="data")

        _session_state["pages"] = result["pages"]
        _session_state["source_name"] = source_name
        n_pages = result["total_pages"]

        logger.info("🔢 Building vector store...")
        n_chunks = build_vector_store(result["pages"], persist_dir="chroma_db")

        _session_state["ingested"] = True
        return (
            f"✅ Ingestion complete!\n\n"
            f"- **Pages processed**: {n_pages}\n"
            f"- **Chunks indexed**: {n_chunks}\n"
            f"- **Source**: `{source_name}`\n\n"
            f"You can now ask questions in the **💬 Chat** tab."
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return f"❌ Ingestion failed: {e}"


def rag_retrieve(question: str, top_k: int, search_mode: str) -> Tuple[List[Dict], str]:
    """
    Retrieve relevant chunks from the vector store.
    Returns (chunks_list, formatted_context_string).
    """
    if not _session_state["ingested"]:
        return [], ""

    try:
        if search_mode == "Semantic":
            chunks = retrieve_chunks(question, top_k=top_k, persist_dir="chroma_db")
        elif search_mode == "Keyword":
            chunks = keyword_search(question, _session_state["pages"], max_results=top_k)
        else:
            # Hybrid
            sem_chunks = retrieve_chunks(question, top_k=top_k // 2 + 1, persist_dir="chroma_db")
            kw_chunks = keyword_search(question, _session_state["pages"], max_results=top_k // 2 + 1)
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


def format_rag_sources(chunks: List[Dict]) -> str:
    """Format retrieved RAG chunks into a readable markdown panel."""
    if not chunks:
        return "*No passages retrieved.*"

    md = ""
    for i, c in enumerate(chunks, 1):
        score = c.get("score", "N/A")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
        text_preview = c["text"][:500]
        if len(c["text"]) > 500:
            text_preview += "..."

        md += f"""**📄 [{i}] Page {c['page_num']}** — _{c['strategy']}_ — Score: `{score_str}`

> {text_preview}

---

"""
    return md.strip()


def load_existing_data() -> str:
    """Try to load pre-ingested data on startup."""
    json_files = list(Path("data").glob("*_pages.json"))
    if not json_files:
        return "No pre-ingested data found. Upload a PDF to begin."

    latest = sorted(json_files)[-1]
    try:
        with open(latest, "r", encoding="utf-8") as f:
            pages = json.load(f)
        _session_state["pages"] = pages
        _session_state["ingested"] = True
        _session_state["source_name"] = latest.stem.replace("_pages", "")

        collection = get_chroma_collection("chroma_db")
        n = collection.count()
        return (
            f"✅ Loaded existing data: `{latest.name}`\n\n"
            f"- **Pages**: {len(pages)}\n"
            f"- **Indexed chunks**: {n}\n\n"
            f"Ready to answer questions!"
        )
    except Exception as e:
        return f"Could not load existing data: {e}"


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------

def chat_respond(user_message: str, history: list, search_mode: str, top_k: int):
    """
    Main chat handler.
    - Retrieves RAG context
    - Generates Groq LLM answer
    - Returns updated chat history + RAG sources panel
    """
    if not user_message.strip():
        return history, ""

    if not _session_state["ingested"]:
        history.append({"role": "user", "content": user_message})
        history.append({
            "role": "assistant",
            "content": "⚠️ No document is loaded yet. Please go to the **📤 Upload & Ingest** tab first to upload a PDF."
        })
        return history, ""

    # Step 1: Retrieve from RAG
    chunks, context = rag_retrieve(user_message, int(top_k), search_mode)

    # Step 2: Generate answer via Groq
    if context:
        answer = answer_with_groq(user_message, context, chat_history=history)
    else:
        answer = "I couldn't find any relevant passages in the document for your question. Please try rephrasing or using a different search mode."

    # Step 3: Format the RAG sources panel
    rag_display = format_rag_sources(chunks)

    # Step 4: Build the assistant response with both answer and RAG indicator
    num_sources = len(chunks)
    source_indicator = f"\n\n---\n*📚 Based on {num_sources} retrieved passage(s) from `{_session_state['source_name']}`*"

    full_answer = answer + source_indicator

    # Update history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": full_answer})

    return history, rag_display


def clear_chat():
    """Clear chat history and RAG panel."""
    return [], ""


# ---------------------------------------------------------------------------
# Gradio UI — Chat Interface
# ---------------------------------------------------------------------------

def build_ui():
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    groq_badge = "✅ Connected" if groq_key_set else "❌ Missing API Key"

    with gr.Blocks(
        title="📖 Multilingual RAG Chat — Groq",
    ) as demo:

        # ── Header ──
        gr.Markdown(f"""
# 📖 Multilingual RAG Chat Assistant
**Bhagavad Gita Knowledge Base** — Ask questions in **English**, **Gujarati** (ગુજરાતી), **Sanskrit** (संस्कृत), or **Hindi** (हिन्दी)

| Component | Status |
|-----------|--------|
| 🤖 LLM | Groq — LLaMA 3.3 70B Versatile |
| 🔌 API | {groq_badge} |
| 📑 Document | `{_session_state.get('source_name', 'None loaded')}` |
        """)

        # ── Tab 1: Chat ──
        with gr.Tab("💬 Chat"):
            with gr.Row():
                # Left: Chat area (70% width)
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=520,
                        layout="bubble",
                        buttons=["copy", "copy_all"],
                        placeholder="Upload a PDF in the '📤 Upload & Ingest' tab, then ask me anything about the Bhagavad Gita! 🙏\n\nYou can ask in English, ગુજરાતી, संस्कृत, or हिन्दी.",
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your question here... (English / ગુજરાતી / संस्कृत / हिन्दी)",
                            label="Your Message",
                            scale=8,
                            lines=1,
                            max_lines=3,
                        )
                        send_btn = gr.Button("📨 Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                        gr.Markdown("*Powered by Groq LLaMA 3.3 70B • ChromaDB • Multilingual Embeddings*", elem_classes=["footer-text"])

                # Right: RAG Sources panel (30% width)
                with gr.Column(scale=3):
                    gr.Markdown("### 📚 Retrieved Context (RAG)")
                    gr.Markdown("*These are the document passages used to answer your question:*")
                    rag_panel = gr.Markdown(
                        value="*Ask a question to see retrieved passages here.*",
                        label="RAG Sources",
                    )
                    with gr.Accordion("⚙️ Search Settings", open=False):
                        search_mode = gr.Radio(
                            choices=["Semantic", "Keyword", "Hybrid"],
                            value="Semantic",
                            label="Search Mode",
                        )
                        top_k = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Passages to retrieve",
                        )

            # Wire up chat events
            msg_input.submit(
                fn=chat_respond,
                inputs=[msg_input, chatbot, search_mode, top_k],
                outputs=[chatbot, rag_panel],
            ).then(lambda: "", outputs=msg_input)

            send_btn.click(
                fn=chat_respond,
                inputs=[msg_input, chatbot, search_mode, top_k],
                outputs=[chatbot, rag_panel],
            ).then(lambda: "", outputs=msg_input)

            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, rag_panel],
            )

            # Example questions
            gr.Examples(
                examples=[
                    ["What is Dharmakshetra?"],
                    ["Who is Duryodhana and what role does he play?"],
                    ["ધર્મક્ષેત્ર નો અર્થ શું છે?"],
                    ["भगवद्गीता का पहला श्लोक क्या है?"],
                    ["What does Bhishma represent in the Gita?"],
                    ["Explain the concept of Guna"],
                ],
                inputs=[msg_input],
                label="💡 Try these questions",
            )

        # ── Tab 2: Upload & Ingest ──
        with gr.Tab("📤 Upload & Ingest"):
            gr.Markdown("### Upload a multilingual PDF for analysis")
            gr.Markdown("Supports **scanned** and **digital** PDFs in English, Gujarati, and Sanskrit.")
            with gr.Row():
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
            ingest_btn = gr.Button("🚀 Ingest & Index Document", variant="primary")
            ingest_status = gr.Markdown(value=load_existing_data())

            ingest_btn.click(
                fn=ingest_and_index,
                inputs=[pdf_input],
                outputs=[ingest_status],
            )

        # ── Tab 3: Document Info ──
        with gr.Tab("📊 Document Info"):
            gr.Markdown("### Document Statistics & Language Analysis")

            def get_stats():
                if not _session_state["ingested"]:
                    return "No document loaded."
                pages = _session_state["pages"]
                collection = get_chroma_collection("chroma_db")
                n_chunks = collection.count()

                langs = {}
                scanned = sum(1 for p in pages if p.get("scanned"))
                for p in pages:
                    l = p.get("language_detected", "unknown")
                    langs[l] = langs.get(l, 0) + 1

                stats = f"""
**Source**: `{_session_state['source_name']}`

| Metric | Value |
|--------|-------|
| Total pages | {len(pages)} |
| Scanned pages (OCR'd) | {scanned} |
| Digital pages | {len(pages) - scanned} |
| Total indexed chunks | {n_chunks} |

**🌐 Detected Languages per page:**
"""
                for lang, count in sorted(langs.items(), key=lambda x: -x[1]):
                    stats += f"\n- `{lang}`: {count} pages"
                return stats

            stats_btn = gr.Button("🔄 Refresh Stats")
            stats_out = gr.Markdown()
            stats_btn.click(fn=get_stats, outputs=stats_out)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        logger.warning("⚠️  GROQ_API_KEY not found in .env — QA will not work until you set it.")
    else:
        logger.info("✅ GROQ_API_KEY loaded successfully.")

    demo = build_ui()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )