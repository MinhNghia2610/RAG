import os

# Thư mục dữ liệu
DATA_DIR = "data/documents"
EMBEDDING_DIR = "data/embeddings"
VECTOR_STORE_PATH = os.path.join(EMBEDDING_DIR, "vector_store.faiss")
METADATA_PATH = os.path.join(EMBEDDING_DIR, "metadata.json")

# Mô hình sử dụng
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

# API Key
OPENAI_API_KEY = os.getenv("AIzaSyAtaS5M2yuqzADb6RxwNracb5HjjII1XPE")
