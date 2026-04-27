"""
MODULE 1: Document Understanding + Preprocessing
Handles PDF reading, OCR detection, multilingual text extraction,
cleaning, and markdown export.
"""

import os
import re
import json
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
import io
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR backends (graceful fallback)
# ---------------------------------------------------------------------------

def _try_import_easyocr():
    try:
        import easyocr
        return easyocr
    except ImportError:
        return None

def _try_import_tesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        return None

def _try_import_langdetect():
    try:
        from langdetect import detect
        return detect
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Page-level helpers
# ---------------------------------------------------------------------------

def is_scanned_page(page: fitz.Page, text_threshold: int = 30) -> bool:
    """Return True if the page has very little selectable text (likely scanned)."""
    text = page.get_text("text").strip()
    return len(text) < text_threshold


def is_garbled_text(text: str) -> bool:
    """
    Detect if extracted text is garbled due to custom non-Unicode font encoding.
    PDFs with custom Gujarati/Sanskrit fonts map native glyphs to Latin char codes,
    producing text full of extended Latin chars (0x80-0xFF) that is not real content.
    
    Returns True if the text appears garbled / not meaningful.
    """
    if not text or len(text) < 10:
        return True
    
    # Count characters in extended Latin range (0x80-0xFF) — hallmark of garbled fonts
    extended_latin = sum(1 for c in text if 0x80 <= ord(c) <= 0xFF)
    # Count proper Unicode Gujarati (U+0A80–U+0AFF) and Devanagari (U+0900–U+097F)
    native_indic = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F or 0x0A80 <= ord(c) <= 0x0AFF)
    # Count standard ASCII printable
    ascii_printable = sum(1 for c in text if 0x20 <= ord(c) <= 0x7E)
    
    total_chars = len(text.replace(' ', '').replace('\n', ''))
    if total_chars == 0:
        return True
    
    extended_ratio = extended_latin / total_chars
    
    # If more than 15% of chars are in extended Latin range (0x80-0xFF),
    # and there are very few proper Indic Unicode chars, it's garbled
    if extended_ratio > 0.15 and native_indic < total_chars * 0.05:
        logger.info(f"  → Garbled text detected: {extended_ratio:.1%} extended Latin, {native_indic} Indic chars")
        return True
    
    return False


def extract_text_native(page: fitz.Page) -> str:
    """Extract native (digital) text from a PDF page."""
    return page.get_text("text").strip()


def page_to_image(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Render a PDF page to a PIL Image for OCR. Uses 300 DPI for better accuracy."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    return Image.open(io.BytesIO(img_bytes))


# ---------------------------------------------------------------------------
# OCR runners
# ---------------------------------------------------------------------------

def ocr_with_easyocr(image: Image.Image, languages: List[str] = None) -> str:
    """Run EasyOCR on a PIL Image. Supports multilingual."""
    easyocr = _try_import_easyocr()
    if easyocr is None:
        raise ImportError("easyocr not installed")
    
    # EasyOCR language codes: 'en' for English, 'gu' doesn't exist natively,
    # but 'hi' (Devanagari) covers Sanskrit; English + Hindi as fallback for Gujarati
    lang_list = languages or ["en", "hi"]
    reader = easyocr.Reader(lang_list, gpu=False, verbose=False)
    
    import numpy as np
    img_array = np.array(image)
    results = reader.readtext(img_array, detail=0, paragraph=True)
    return "\n".join(results)


def ocr_with_tesseract(image: Image.Image, lang: str = "eng+guj+san") -> str:
    """Run Tesseract OCR on a PIL Image with multilingual support."""
    pytesseract = _try_import_tesseract()
    if pytesseract is None:
        raise ImportError("pytesseract not installed")
    
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        # Fallback to English only
        logger.warning(f"Tesseract with lang={lang} failed ({e}), retrying with 'eng'")
        return pytesseract.image_to_string(image, lang="eng").strip()


def ocr_page(image: Image.Image) -> str:
    """Try EasyOCR first, fall back to Tesseract, then empty string."""
    # Attempt 1: EasyOCR
    try:
        text = ocr_with_easyocr(image, languages=["en", "hi"])
        if text.strip():
            logger.info("  → OCR via EasyOCR succeeded")
            return text
    except Exception as e:
        logger.warning(f"  EasyOCR failed: {e}")

    # Attempt 2: Tesseract
    try:
        text = ocr_with_tesseract(image)
        if text.strip():
            logger.info("  → OCR via Tesseract succeeded")
            return text
    except Exception as e:
        logger.warning(f"  Tesseract failed: {e}")

    logger.warning("  → Both OCR backends failed; returning empty string")
    return ""


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean noisy OCR / scanned text:
    - Remove control chars
    - Normalize whitespace
    - Remove isolated single chars (common OCR noise)
    - Preserve Gujarati / Devanagari Unicode ranges
    """
    if not text:
        return ""

    # Remove null bytes and other control characters (keep newlines/tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize multiple spaces to single space (but keep newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove lines that are only punctuation/noise
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep line if it has at least 3 meaningful characters
        meaningful = re.sub(r"[^\w\u0A80-\u0AFF\u0900-\u097F]", "", stripped)
        if len(meaningful) >= 2:
            cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines).strip()


def detect_language(text: str) -> str:
    """Detect language of text; return ISO code or 'unknown'."""
    detect = _try_import_langdetect()
    if detect is None or not text.strip():
        return "unknown"
    try:
        return detect(text[:500])
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Structure inference
# ---------------------------------------------------------------------------

def infer_heading(line: str, page_num: int) -> Optional[str]:
    """
    Heuristically detect headings:
    - Short lines (< 80 chars) that start with capital letter or Devanagari/Gujarati char
    - Lines that look like chapter/verse markers
    """
    line = line.strip()
    if not line:
        return None
    
    # Sanskrit shloka markers (double pipe JJ, colon patterns)
    if re.search(r"JJ\d+JJ|॥\d+॥|\|\|\d+\|\|", line):
        return "verse"
    
    # Very short, likely a heading
    if len(line) < 80 and re.match(r"^[A-Z\u0A85-\u0ABE\u0900-\u093F]", line):
        return "heading"
    
    return None


def structure_page_text(raw_text: str, page_num: int) -> str:
    """Convert raw text into markdown-like structured output."""
    if not raw_text.strip():
        return ""

    lines = raw_text.split("\n")
    structured_lines = [f"\n---\n## Page {page_num}\n"]

    for line in lines:
        if not line.strip():
            continue
        htype = infer_heading(line, page_num)
        if htype == "verse":
            structured_lines.append(f"\n> {line}\n")
        elif htype == "heading":
            structured_lines.append(f"\n### {line}\n")
        else:
            structured_lines.append(line)

    return "\n".join(structured_lines)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: str,
    output_dir: str = "data",
    dpi: int = 200,
) -> Dict:
    """
    Full ingestion pipeline:
    1. Open PDF
    2. Per page: detect if scanned → OCR, else extract natively
    3. Clean and structure text
    4. Export as markdown and JSON
    
    Returns a dict with metadata and list of page dicts.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Opening PDF: {pdf_path.name}")
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"Total pages: {total_pages}")

    pages = []
    full_markdown = f"# {pdf_path.stem}\n\n"

    for page_num in range(total_pages):
        page = doc[page_num]
        pg_label = page_num + 1
        logger.info(f"Processing page {pg_label}/{total_pages} ...")

        scanned = is_scanned_page(page)
        used_ocr = False
        
        if scanned:
            # Clearly scanned: no selectable text → OCR
            logger.info(f"  Page {pg_label}: scanned → running OCR")
            img = page_to_image(page, dpi=dpi)
            raw_text = ocr_page(img)
            used_ocr = True
        else:
            # Has selectable text — but check if it's garbled
            native_text = extract_text_native(page)
            
            if is_garbled_text(native_text):
                # Garbled font encoding detected → force OCR
                logger.info(f"  Page {pg_label}: garbled font encoding detected → forcing OCR")
                img = page_to_image(page, dpi=dpi)
                raw_text = ocr_page(img)
                used_ocr = True
                scanned = True  # Mark as effectively scanned
            else:
                logger.info(f"  Page {pg_label}: clean digital text → native extraction")
                raw_text = native_text

        cleaned = clean_text(raw_text)
        lang = detect_language(cleaned)
        structured = structure_page_text(cleaned, pg_label)

        page_data = {
            "page_num": pg_label,
            "scanned": scanned,
            "language_detected": lang,
            "raw_text": raw_text,
            "cleaned_text": cleaned,
            "structured_md": structured,
            "used_ocr": used_ocr,
        }
        pages.append(page_data)
        full_markdown += structured + "\n"

    doc.close()

    # Save outputs
    md_path = output_dir / f"{pdf_path.stem}_extracted.md"
    json_path = output_dir / f"{pdf_path.stem}_pages.json"

    md_path.write_text(full_markdown, encoding="utf-8")
    logger.info(f"Markdown saved → {md_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved → {json_path}")

    result = {
        "source": str(pdf_path),
        "total_pages": total_pages,
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "pages": pages,
    }
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf> [output_dir]")
        sys.exit(1)
    
    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data"
    result = ingest_pdf(pdf, output_dir=out)
    print(f"\nDone. Pages processed: {result['total_pages']}")
    print(f"Markdown: {result['markdown_path']}")
    print(f"JSON:     {result['json_path']}")