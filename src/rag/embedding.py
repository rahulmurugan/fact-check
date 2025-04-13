

import os
import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer



def load_records(pdf_folder: Path) -> List[Dict]:
    """Collect paragraph, table, and figure‑OCR records."""
    records = []
    for fname in ("text_chunks.jsonl", "tables.jsonl", "image_ocr.jsonl"):
        fpath = pdf_folder / fname
        if not fpath.exists():
            continue
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                rec["source_pdf"] = pdf_folder.name  # add provenance
                records.append(rec)
    return records



def build_corpus(data_dir: Path) -> List[Dict]:
    corpus = []
    for pdf_dir in sorted(data_dir.iterdir()):
        if pdf_dir.is_dir():
            corpus.extend(load_records(pdf_dir))
    print(f" Loaded {len(corpus):,} records from {data_dir}")
    return corpus


def create_embeddings(
        texts: list[str],
        model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2",
    ) -> np.ndarray:
    """
    Encode every text one‑by‑one to minimise GPU / RAM spikes.
    Returns a (N, dim) float32 matrix with unit‑length rows.
    """
    model = SentenceTransformer(model_name)          # loads once
    vecs  = []

    for txt in tqdm(texts, desc="🔗 embedding"):
        emb = model.encode(txt, normalize_embeddings=True)
        vecs.append(emb)

    return np.vstack(vecs).astype("float32")

def build_faiss_index(embeddings, metadata: List[Dict], vdb_dir: Path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # cosine similarity (embs already normalized)
    index.add(embeddings)

    vdb_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(vdb_dir / "index.faiss"))
    with (vdb_dir / "docstore.pkl").open("wb") as f:
        pickle.dump(metadata, f)

    print(f"✅  Stored {len(metadata):,} vectors to {vdb_dir}")