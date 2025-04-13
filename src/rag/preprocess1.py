
import os
import json
import uuid
from pathlib import Path
from typing import List, Dict

import fitz                              # PyMuPDF  image & text
import pytesseract                       # OCR
from PIL import Image                    # image I/O for OCR
import camelot                           # table extraction
from nltk.tokenize import sent_tokenize  # basic text splitting
import re
import warnings
import pytesseract
from pytesseract import TesseractError



def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = text.strip()
    return text

def _split_into_paragraphs(page_text: str) -> List[str]:
    raw_paragraphs = page_text.split('\n')
    return [_clean_text(p) for p in raw_paragraphs if p.strip()]



def extract_text(pdf_path: Path, out_dir: Path) -> None:
    """Extract & split plain text into JSONL."""
    doc = fitz.open(pdf_path)
    out_file = out_dir / "text_chunks.jsonl"
    with out_file.open("w", encoding="utf‑8") as f:
        for page_num, page in enumerate(doc, start=1):
            for para in _split_into_paragraphs(page.get_text("text")):
                if para:
                    json.dump(
                        {
                            "id": str(uuid.uuid4()),
                            "page": page_num,
                            "type": "paragraph",
                            "text": para,
                        },
                        f,
                    )
                    f.write("\n")
    doc.close()


def _nonzero_area(t):
    x1, y1, x2, y2 = map(float, t._bbox)   # (left, top, right, bottom)
    return (x2 - x1) * (y1 - y2) > 0       # stream flavour uses PDF coords

def extract_tables(pdf_path: Path, out_dir: Path) -> None:
    """
    Extract tables with Camelot; skip zero‑area artefacts that crash later code.
    """
    out_file = out_dir / "tables.jsonl"

    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all",
                                  flavor="stream", suppress_stdout=True)
    except Exception as e:
        print(f" Camelot failed on {pdf_path.name}: {e}")
        return                                     # nothing to write

    valid_tables = [t for t in tables if _nonzero_area(t)]

    with out_file.open("w", encoding="utf-8") as f:
        for t in valid_tables:
            json.dump(
                {
                    "id": str(uuid.uuid4()),
                    "pdf": pdf_path.name,
                    "bbox": t._bbox,              # or anything else you need
                    "data": t.df.fillna("").values.tolist(),
                },
                f,
            )
            f.write("\n")

    print(f"📝  {len(valid_tables)} table(s) written for {pdf_path.name}")


def _safe_ocr(img_path: Path) -> str:
    """Run Tesseract but never crash the pipeline."""
    try:
        return pytesseract.image_to_string(Image.open(img_path))
    except (AssertionError, TesseractError, OSError) as e:
        warnings.warn(f"OCR skipped for {img_path.name}: {e}")
        return ""                       # empty string = no OCR text

def extract_images_and_ocr(pdf_path: Path, out_dir: Path) -> None:
    img_dir   = out_dir / "images"
    img_dir.mkdir(exist_ok=True, parents=True)
    ocr_file  = out_dir / "image_ocr.jsonl"

    doc = fitz.open(pdf_path)
    with ocr_file.open("w", encoding="utf-8") as f:
        for page_num, page in enumerate(doc, 1):
            for img_idx, (xref, *_) in enumerate(page.get_images(full=True)):
                pix = fitz.Pixmap(doc, xref)

                # skip 0‑area or grayscale mask pixmaps
                if pix.width == 0 or pix.height == 0 or pix.n < 3:
                    pix = None
                    continue

                img_path = img_dir / f"fig_{page_num:03}_{img_idx:02}.png"
                pix.save(img_path)
                pix = None                           # free RAM

                text = _safe_ocr(img_path)
                json.dump(
                    {
                        "id":   str(uuid.uuid4()),
                        "page": page_num,
                        "type": "figure",
                        "file": str(img_path),
                        "ocr_text": text.strip(),
                    },
                    f,
                )
                f.write("\n")
    doc.close()
