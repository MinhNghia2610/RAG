# utils.py
# Hàm tiện ích: load/save metadata, embed text, tách đoạn, xử lý text cơ bản

import json
import os
import re
from typing import List
from pathlib import Path

from config import METADATA_PATH, EMBEDDING_MODEL_NAME

# embedding: dùng sentence-transformers nếu có
def get_sentence_transformer_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Không tìm thấy thư viện sentence-transformers. "
            "Cài bằng: pip install -U sentence-transformers"
        ) from e
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

def embed_texts(texts: List[str]):
    """
    Trả về numpy array (n, dim) embedding cho danh sách texts.
    Dùng sentence-transformers local.
    """
    model = get_sentence_transformer_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

def normalize_text(s: str) -> str:
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def load_metadata(path: Path = METADATA_PATH):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_metadata(meta, path: Path = METADATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def chunk_text(text: str, max_chars: int = 800):
    """
    Chia text lớn thành các đoạn nhỏ ~ max_chars (cố gắng ngắt câu).
    """
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    parts = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                parts.append(cur)
            cur = s
    if cur:
        parts.append(cur)
    return parts
