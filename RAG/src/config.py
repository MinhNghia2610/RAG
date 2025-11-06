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
OPENAI_API_KEY = os.getenv("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")
