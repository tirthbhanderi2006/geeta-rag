"""
MODULE 2: Chunking + Embedding Strategy
Implements multiple chunking strategies and stores embeddings in ChromaDB.
Uses multilingual sentence-transformers model.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Multilingual embedding model
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_structure(pages: List[Dict], max_chunk_size: int = 600) -> List[Dict]:
    """
    Strategy 1: Structure-based chunking.
    Split on markdown headings (##, ###) and verse markers (>).
    Respects page boundaries.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page.get("cleaned_text", "")
        if not text.strip():
            continue

        # Split on heading-like patterns
        sections = re.split(r"\n(?=#{1,3} |\> )", text)
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # If section is within size limit, add as-is
            if len(section) <= max_chunk_size:
                chunks.append({
                    "chunk_id": f"struct_{chunk_id}",
                    "text": section,
                    "page_num": page["page_num"],
                    "strategy": "structure",
                    "language": page.get("language_detected", "unknown"),
                })
                chunk_id += 1
            else:
                # Sub-split long sections by sentence
                sub_chunks = _split_by_sentences(section, max_chunk_size)
                for sub in sub_chunks:
                    chunks.append({
                        "chunk_id": f"struct_{chunk_id}",
                        "text": sub,
                        "page_num": page["page_num"],
                        "strategy": "structure_sub",
                        "language": page.get("language_detected", "unknown"),
                    })
                    chunk_id += 1

    logger.info(f"Structure chunking → {len(chunks)} chunks")
    return chunks


def chunk_recursive(pages: List[Dict], chunk_size: int = 500, overlap: int = 80) -> List[Dict]:
    """
    Strategy 2: Recursive character-level chunking with overlap.
    Good for dense text where structure is unclear.
    """
    # Combine all text with page markers
    all_text_parts = []
    for page in pages:
        text = page.get("cleaned_text", "")
        if text.strip():
            all_text_parts.append((page["page_num"], text))

    chunks = []
    chunk_id = 0

    for page_num, text in all_text_parts:
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Try to end at a sentence boundary
            if end < text_len:
                boundary = _find_sentence_boundary(text, start, end)
                end = boundary if boundary > start else end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"recur_{chunk_id}",
                    "text": chunk_text,
                    "page_num": page_num,
                    "strategy": "recursive",
                    "language": "unknown",
                })
                chunk_id += 1

            start = max(start + 1, end - overlap)

    logger.info(f"Recursive chunking → {len(chunks)} chunks")
    return chunks


def chunk_with_overlap(pages: List[Dict], chunk_size: int = 400, overlap: int = 100) -> List[Dict]:
    """
    Strategy 3: Sliding window with overlap, page-aware.
    Ensures context continuity across chunk boundaries.
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page.get("cleaned_text", "")
        if not text.strip():
            continue

        words = text.split()
        if not words:
            continue

        # Approximate word-based chunking (more language-agnostic)
        words_per_chunk = chunk_size // 5  # ~5 chars/word avg
        overlap_words = overlap // 5
        start = 0

        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": f"overlap_{chunk_id}",
                    "text": chunk_text,
                    "page_num": page["page_num"],
                    "strategy": "overlap",
                    "language": page.get("language_detected", "unknown"),
                })
                chunk_id += 1
            start += max(1, words_per_chunk - overlap_words)

    logger.info(f"Overlap chunking → {len(chunks)} chunks")
    return chunks


def _split_by_sentences(text: str, max_size: int) -> List[str]:
    """Split text into sub-chunks at sentence boundaries."""
    # Split on common sentence endings; works for English and partially for Gujarati
    sentences = re.split(r"(?<=[.।?!])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_size]]


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find the last sentence boundary before 'end'."""
    for i in range(end, max(start, end - 100), -1):
        if text[i] in ".।?!\n":
            return i + 1
    return end


def merge_and_dedupe(all_chunks: List[Dict]) -> List[Dict]:
    """Remove duplicate chunks (same text, different strategy)."""
    seen = set()
    unique = []
    for c in all_chunks:
        key = c["text"].strip()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    logger.info(f"After deduplication: {len(unique)} unique chunks")
    return unique


# ---------------------------------------------------------------------------
# Embedding + ChromaDB
# ---------------------------------------------------------------------------

def get_embedding_model():
    """Load the multilingual sentence-transformer model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    return model


def get_chroma_collection(persist_dir: str = CHROMA_DIR):
    """Initialize or load a ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_and_store(
    chunks: List[Dict],
    persist_dir: str = CHROMA_DIR,
    batch_size: int = 64,
) -> None:
    """
    Embed all chunks using the multilingual model and store in ChromaDB.
    Processes in batches to avoid OOM on large documents.
    """
    if not chunks:
        logger.warning("No chunks to embed.")
        return

    model = get_embedding_model()
    collection = get_chroma_collection(persist_dir)

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "page_num": c["page_num"],
            "strategy": c["strategy"],
            "language": c.get("language", "unknown"),
        }
        for c in chunks
    ]

    logger.info(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")
    
    for i in range(0, len(chunks), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]

        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        logger.info(f"  Stored batch {i // batch_size + 1} ({len(batch_texts)} chunks)")

    logger.info(f"✓ Stored {len(chunks)} chunks in ChromaDB at '{persist_dir}'")


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    persist_dir: str = CHROMA_DIR,
) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query using semantic similarity.
    Also supports keyword fallback if semantic results are poor.
    """
    model = get_embedding_model()
    collection = get_chroma_collection(persist_dir)

    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty. Run ingestion first.")
        return []

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "page_num": meta.get("page_num", "?"),
            "strategy": meta.get("strategy", "?"),
            "language": meta.get("language", "?"),
            "score": round(1 - dist, 4),  # cosine similarity
        })

    return chunks


def keyword_search(
    keyword: str,
    pages: List[Dict],
    max_results: int = 5,
) -> List[Dict]:
    """Simple keyword search over page texts as fallback."""
    keyword_lower = keyword.lower()
    results = []
    for page in pages:
        text = page.get("cleaned_text", "")
        if keyword_lower in text.lower():
            # Extract surrounding context
            idx = text.lower().find(keyword_lower)
            start = max(0, idx - 150)
            end = min(len(text), idx + 300)
            snippet = text[start:end].strip()
            results.append({
                "text": snippet,
                "page_num": page["page_num"],
                "strategy": "keyword",
                "language": page.get("language_detected", "?"),
                "score": 1.0,
            })
            if len(results) >= max_results:
                break
    return results


# ---------------------------------------------------------------------------
# Full pipeline runner
# ---------------------------------------------------------------------------

def build_vector_store(pages: List[Dict], persist_dir: str = CHROMA_DIR) -> int:
    """
    Run all three chunking strategies, merge, deduplicate, embed, and store.
    Returns total number of chunks stored.
    """
    chunks_struct = chunk_by_structure(pages)
    chunks_recur = chunk_recursive(pages)
    chunks_overlap = chunk_with_overlap(pages)

    all_chunks = chunks_struct + chunks_recur + chunks_overlap
    unique_chunks = merge_and_dedupe(all_chunks)
    
    embed_and_store(unique_chunks, persist_dir=persist_dir)
    return len(unique_chunks)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <pages_json_path> [chroma_dir]")
        sys.exit(1)
    
    json_path = sys.argv[1]
    chroma_dir = sys.argv[2] if len(sys.argv) > 2 else CHROMA_DIR

    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    
    total = build_vector_store(pages, persist_dir=chroma_dir)
    print(f"\nDone. {total} chunks stored in ChromaDB at '{chroma_dir}'")