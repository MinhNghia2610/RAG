# config.py
# Cấu hình chung của dự án

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # thư mục RAG
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"

# đường dẫn file vector store và metadata
FAISS_INDEX_PATH = EMB_DIR / "vector_store.faiss"
METADATA_PATH = EMB_DIR / "metadata.json"

# embedding model config (sử dụng sentence-transformers local)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Số document trả về mặc định
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

# Tùy chọn: nếu muốn dùng OpenAI cho generation (không bắt buộc)
USE_OPENAI_FOR_GENERATION = os.getenv("USE_OPENAI_FOR_GENERATION", "false").lower() in ("1","true","yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # thay đổi nếu cần
