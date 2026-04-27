# 📖 Multilingual Knowledge Extraction & Exploration Assistant

A complete RAG (Retrieval-Augmented Generation) pipeline for multilingual PDF documents — specifically designed for scanned/OCR'd handwritten texts in **Gujarati, Sanskrit, and English** (e.g., Yatharth Geeta).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      USER (Gradio UI)                       │
│                  Question / Upload PDF                       │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼───────────────┐
          │     MODULE 1: INGEST         │
          │  ingest.py                   │
          │  ┌─────────┐  ┌──────────┐  │
          │  │ PyMuPDF │  │ pdfplumb │  │
          │  └────┬────┘  └────┬─────┘  │
          │       │            │         │
          │  is_scanned?       │         │
          │  YES→OCR    NO→native        │
          │  ┌─────────────────────┐     │
          │  │ EasyOCR (primary)   │     │
          │  │ Tesseract (fallback)│     │
          │  └──────────┬──────────┘     │
          │             │                │
          │         clean_text()         │
          │         detect_lang()        │
          │         structure_md()       │
          └──────────────┬───────────────┘
                         │
                    pages.json
                    extracted.md
                         │
          ┌──────────────▼───────────────┐
          │     MODULE 2: CHUNK+EMBED    │
          │  utils.py                    │
          │                              │
          │  3 Chunking Strategies:      │
          │  ┌──────────┐               │
          │  │Structure │ → by headings  │
          │  ├──────────┤               │
          │  │Recursive │ → char-based   │
          │  ├──────────┤               │
          │  │Overlap   │ → sliding win  │
          │  └────┬─────┘               │
          │       │ deduplicate          │
          │       ▼                      │
          │  sentence-transformers       │
          │  paraphrase-multilingual-    │
          │  MiniLM-L12-v2              │
          │       │ embed                │
          │       ▼                      │
          │    ChromaDB (cosine)         │
          └──────────────┬───────────────┘
                         │
          ┌──────────────▼───────────────┐
          │     MODULE 3: QA / EXPLORE   │
          │  app.py                      │
          │                              │
          │  Query → encode → retrieve   │
          │  top-k chunks                │
          │       │                      │
          │  Build RAG prompt            │
          │       │                      │
          │  LLM Backend (modular):      │
          │  ┌────────────────────────┐  │
          │  │ Extractive (offline)   │  │
          │  │ HF local (flan-t5)     │  │
          │  │ OpenAI GPT-3.5         │  │
          │  │ Claude (Anthropic)     │  │
          │  └────────────────────────┘  │
          │       │                      │
          │  Answer + Sources → UI       │
          └──────────────────────────────┘
```

---

## ⚡ Quick Start (45-minute setup)

### 1. Clone / create project folder

```bash
mkdir rag_project && cd rag_project
# copy all files here
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tesseract OCR** (optional, for fallback):
> ```bash
> # Ubuntu/Debian
> sudo apt-get install tesseract-ocr tesseract-ocr-guj tesseract-ocr-san
> 
> # macOS
> brew install tesseract
> ```

### 3. Set up API keys (optional)

```bash
cp .env.example .env
# Edit .env and add your keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

> Without API keys, the **extractive QA backend** works fully offline.

### 4. Run the full pipeline

**Option A: All-in-one (ingest + UI)**
```bash
python app.py
```
Then open http://localhost:7860 in your browser.

**Option B: Step by step**
```bash
# Step 1: Ingest PDF
python ingest.py your_book.pdf data/

# Step 2: Build vector store
python utils.py data/your_book_pages.json chroma_db/

# Step 3: Launch UI
python app.py
```

---

## 📂 Project Structure

```
rag_project/
├── app.py              # Gradio UI + LLM routing (Module 3)
├── ingest.py           # PDF ingestion + OCR pipeline (Module 1)
├── utils.py            # Chunking + embedding + ChromaDB (Module 2)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── .env.example        # API key template
├── data/               # Extracted text files (auto-created)
│   ├── *_extracted.md  # Structured markdown
│   └── *_pages.json    # Per-page data with metadata
└── chroma_db/          # ChromaDB vector store (auto-created)
```

---

## 🔧 Models & Tools Used

| Component | Tool / Model | Reason |
|-----------|-------------|--------|
| PDF reading | PyMuPDF (fitz) | Fast, handles both digital & scanned |
| OCR (primary) | EasyOCR | Multilingual, no Tesseract install needed |
| OCR (fallback) | Tesseract + pytesseract | Gujarati lang pack available |
| Text cleaning | Custom regex | Handles Unicode ranges for Gujarati/Devanagari |
| Language detection | langdetect | Lightweight, per-page detection |
| Chunking | 3-strategy custom | Structure + Recursive + Overlap |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2 | 50+ languages, fast, free |
| Vector store | ChromaDB | Local, persistent, cosine similarity |
| Extractive QA | deepset/roberta-base-squad2 | Offline, no API key needed |
| Generative QA | flan-t5-base / GPT-3.5 / Claude | Modular, plug any backend |
| UI | Gradio | Rapid prototyping, file upload built-in |

---

## 🌐 Multilingual Handling

### Gujarati Support
- **EasyOCR**: Uses `["en", "hi"]` language codes (Hindi shares Devanagari script; Gujarati has its own script but EasyOCR handles it partially via the Hindi model)
- **Tesseract**: Uses `eng+guj+san` language codes when `tesseract-ocr-guj` pack is installed
- **Embedding**: `paraphrase-multilingual-MiniLM-L12-v2` is trained on 50+ languages including Gujarati

### Sanskrit Support
- Sanskrit uses Devanagari script, handled by Tesseract's `san` language pack
- EasyOCR handles Devanagari via `hi` (Hindi) model

### Noisy Handwriting
- Preprocessing: Convert page to 200 DPI image before OCR
- Cleaning: Remove isolated characters, normalize Unicode, preserve Gujarati range (U+0A80–U+0AFF)
- Fallback chain: EasyOCR → Tesseract → empty string (never crashes)

---

## ✅ What Worked Well

1. **PyMuPDF scan detection** — accurate, fast heuristic using text character count
2. **EasyOCR** — handled the Gujarati text surprisingly well with Hindi model fallback  
3. **Multilingual embeddings** — semantic search worked cross-lingually (English query finds Gujarati passage)
4. **3-strategy chunking** — different strategies captured different granularities of meaning
5. **ChromaDB** — zero-config local vector store, instant setup
6. **Gradio** — UI built in < 100 lines, file upload + tab layout out of the box

---

## ❌ What Didn't Work / Limitations

1. **Gujarati OCR accuracy** — EasyOCR's Gujarati support is incomplete; dedicated Gujarati model would improve accuracy significantly
2. **Sanskrit shloka structure** — verse boundaries not perfectly detected; would need dedicated Sanskrit NLP
3. **Handwritten content** — heavily handwritten pages produce very noisy OCR; transformer-based vision models (like TrOCR) would help
4. **Extractive QA with multilingual** — roberta-base-squad2 is English-only; multilingual QA model (mBERT-based) would be better
5. **Context window** — long documents may exceed LLM context; would need smarter chunk selection
6. **No re-ranking** — chunks returned by cosine similarity alone; cross-encoder re-ranking would improve precision

---

## 🚀 Future Improvements

| Priority | Improvement |
|----------|-------------|
| High | Use `microsoft/trocr-base-handwritten` for handwritten OCR |
| High | Swap to `intfloat/multilingual-e5-large` for better multilingual embeddings |
| High | Add cross-encoder re-ranking (e.g., `cross-encoder/ms-marco-MiniLM-L6-v2`) |
| Medium | Dedicated Gujarati LLM or fine-tuned multilingual model |
| Medium | Hierarchical indexing (chapter → section → chunk) |
| Medium | Streaming answers in UI |
| Low | Export Q&A pairs as dataset |
| Low | Add voice input/output for accessibility |

---

## 🔑 .env Template

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## 📝 Design Decisions

**Why 3 chunking strategies?** Different texts benefit from different granularities. Verse-heavy Sanskrit benefits from structure-based; prose benefits from recursive; and overlap ensures no information is cut at boundaries.

**Why EasyOCR over Tesseract as primary?** EasyOCR requires no system install, handles more scripts out-of-the-box, and is more robust to font variations.

**Why ChromaDB over FAISS/Pinecone?** ChromaDB is persistent, requires no API key, runs locally, and has a simple Python API — ideal for a hackathon prototype.

**Why extractive QA as default?** It works completely offline, requires no API key, and for factual questions about a specific document, extractive answers are often more accurate than generative ones.