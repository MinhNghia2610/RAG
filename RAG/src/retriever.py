# retriever.py
# Tải faiss index + metadata, thực hiện truy vấn nearest neighbors

from config import FAISS_INDEX_PATH, METADATA_PATH, DEFAULT_TOP_K
from utils import load_metadata, embed_texts, normalize_text
from pathlib import Path
import numpy as np

class Retriever:
    def __init__(self, index_path: Path = FAISS_INDEX_PATH, meta_path: Path = METADATA_PATH):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.metadata = load_metadata(self.meta_path)
        self.index = None
        self.dim = None
        self._load_index()

    def _load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"Không tìm thấy FAISS index tại {self.index_path}. Bạn có thể chạy build_embeddings.py để tạo.")
        try:
            import faiss
        except Exception as e:
            raise RuntimeError("Cần faiss (faiss-cpu) để chạy retriever. pip install faiss-cpu") from e
        self.index = faiss.read_index(str(self.index_path))
        # dim có thể lấy từ index.d
        if hasattr(self.index, "d"):
            self.dim = int(self.index.d)
        else:
            # thử đo bằng reconstruct (nếu hỗ trợ)
            self.dim = None

    def _embed_query(self, query: str):
        q = normalize_text(query)
        emb = embed_texts([q])
        # normalize as index built with normalized vectors
        try:
            import faiss
            faiss.normalize_L2(emb)
        except Exception:
            pass
        return emb.astype("float32")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K):
        """
        Trả về danh sách kết quả: [{"score":..., "text":..., "source":...}, ...]
        """
        if self.index is None:
            raise RuntimeError("Index chưa được load.")
        emb = self._embed_query(query)
        D, I = self.index.search(emb, top_k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append({
                "score": float(score),
                "text": m.get("text", ""),
                "source": m.get("source", "")
            })
        return results

# helper function
_default_retriever = None
def get_default_retriever():
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever

def retrieve(query: str, top_k: int = DEFAULT_TOP_K):
    r = get_default_retriever()
    return r.retrieve(query, top_k=top_k)
