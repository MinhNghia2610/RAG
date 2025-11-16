# build_embeddings.py
# Tạo lại vector store FAISS từ folder data (nếu bạn muốn rebuild).
# Sử dụng sentence-transformers để tạo embeddings.

import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import DATA_DIR, EMB_DIR, FAISS_INDEX_PATH, METADATA_PATH
from utils import chunk_text, embed_texts, normalize_text, save_metadata

def collect_documents(data_dir: Path):
    docs = []
    for p in data_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
            docs.append(p)
    return docs

def load_text_file(path: Path):
    if path.suffix.lower() == ".pdf":
        # nhẹ: thử import PyPDF2 nếu có
        try:
            import PyPDF2
        except Exception:
            print(f"[WARN] Cần PyPDF2 để đọc PDF: {path}. Bỏ qua file này.")
            return ""
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def build_index(save_faiss: Path = FAISS_INDEX_PATH, save_meta: Path = METADATA_PATH):
    docs = collect_documents(DATA_DIR)
    if not docs:
        print("Không tìm thấy tài liệu trong data/. Hãy thêm file.")
        return

    all_chunks = []
    meta = []
    for p in docs:
        text = load_text_file(p)
        if not text:
            continue
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "source": str(p.relative_to(Path(__file__).resolve().parents[1])),
                "chunk_id": len(meta),
                "text": ch
            })

    if not all_chunks:
        print("Không có đoạn văn để tạo embedding.")
        return

    # tạo embedding
    print("Tạo embedding cho các đoạn... (cần sentence-transformers)")
    embeddings = embed_texts(all_chunks)  # numpy array

    dim = embeddings.shape[1]

    # tạo faiss index
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("Cần faiss (faiss-cpu) để build index. pip install faiss-cpu") from e

    index = faiss.IndexFlatIP(dim)  # dot-product (ưu tiên normalize trước)
    # normalize vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))
    save_meta(meta, save_meta)
    save_faiss.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_faiss))
    print(f"Đã lưu index vào {save_faiss} và metadata vào {save_meta} (số chunks: {len(meta)})")

if __name__ == "__main__":
    build_index()
# build_embeddings.py
# Tạo lại vector store FAISS từ folder data (nếu bạn muốn rebuild).
# Sử dụng sentence-transformers để tạo embeddings.

import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import DATA_DIR, EMB_DIR, FAISS_INDEX_PATH, METADATA_PATH
from utils import chunk_text, embed_texts, normalize_text, save_metadata

def collect_documents(data_dir: Path):
    docs = []
    for p in data_dir.glob("**/*"):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
            docs.append(p)
    return docs

def load_text_file(path: Path):
    if path.suffix.lower() == ".pdf":
        # nhẹ: thử import PyPDF2 nếu có
        try:
            import PyPDF2
        except Exception:
            print(f"[WARN] Cần PyPDF2 để đọc PDF: {path}. Bỏ qua file này.")
            return ""
        text = ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    else:
        return path.read_text(encoding="utf-8", errors="ignore")

def build_index(save_faiss: Path = FAISS_INDEX_PATH, save_meta: Path = METADATA_PATH):
    docs = collect_documents(DATA_DIR)
    if not docs:
        print("Không tìm thấy tài liệu trong data/. Hãy thêm file.")
        return

    all_chunks = []
    meta = []
    for p in docs:
        text = load_text_file(p)
        if not text:
            continue
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            meta.append({
                "source": str(p.relative_to(Path(__file__).resolve().parents[1])),
                "chunk_id": len(meta),
                "text": ch
            })

    if not all_chunks:
        print("Không có đoạn văn để tạo embedding.")
        return

    # tạo embedding
    print("Tạo embedding cho các đoạn... (cần sentence-transformers)")
    embeddings = embed_texts(all_chunks)  # numpy array

    dim = embeddings.shape[1]

    # tạo faiss index
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("Cần faiss (faiss-cpu) để build index. pip install faiss-cpu") from e

    index = faiss.IndexFlatIP(dim)  # dot-product (ưu tiên normalize trước)
    # normalize vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype("float32"))
    save_meta(meta, save_meta)
    save_faiss.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(save_faiss))
    print(f"Đã lưu index vào {save_faiss} và metadata vào {save_meta} (số chunks: {len(meta)})")

if __name__ == "__main__":
    build_index()
