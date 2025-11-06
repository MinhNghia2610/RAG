# ============================================
# IMPORT CÁC THƯ VIỆN CẦN THIẾT
# ============================================

import json                     # Dùng để đọc và ghi file JSON (metadata)
import faiss                    # Thư viện FAISS (Facebook AI Similarity Search) để tìm kiếm vector hiệu quả
import numpy as np              # Xử lý ma trận và vector số
from openai import OpenAI       # Thư viện tương tác với API OpenAI
from src.config import *        # Import các biến cấu hình từ file config (API key, model, đường dẫn,...)


# ============================================
# KHỞI TẠO CLIENT OPENAI
# ============================================

# Tạo client OpenAI sử dụng API key đã lưu trong file config
client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================
# HÀM LẤY EMBEDDING TỪ MỘT ĐOẠN VĂN BẢN
# ============================================

def get_embedding(text):
    """
    Hàm này nhận vào một đoạn text và trả về vector embedding của nó.

    Args:
        text (str): Chuỗi văn bản cần tạo embedding.

    Returns:
        np.array: Vector embedding có kiểu float32.
    """

    # Gửi yêu cầu tới API của OpenAI để tạo embedding cho đoạn văn bản
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,  # Model embedding (ví dụ: "text-embedding-3-large")
        input=text               # Văn bản cần mã hóa
    )

    # Trích xuất vector embedding từ phản hồi API và chuyển sang numpy array
    return np.array(response.data[0].embedding, dtype=np.float32)


# ============================================
# HÀM TRUY XUẤT THÔNG TIN TỪ VECTOR STORE
# ============================================

def retrieve(query, top_k=3):
    """
    Hàm này thực hiện truy vấn (query) lên kho dữ liệu FAISS để tìm ra
    các đoạn văn bản gần nhất (theo nghĩa ngữ nghĩa) với câu hỏi.

    Args:
        query (str): Câu hỏi hoặc truy vấn của người dùng.
        top_k (int): Số lượng kết quả tương đồng nhất cần lấy ra (mặc định: 3).

    Returns:
        list: Danh sách metadata tương ứng với các đoạn văn bản tìm được.
    """

    # Đọc lại index FAISS từ file đã lưu (VECTOR_STORE_PATH)
    index = faiss.read_index(VECTOR_STORE_PATH)

    # Đọc file metadata chứa thông tin các đoạn văn bản gốc (metadata.json)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Tạo embedding cho câu query người dùng
    query_vec = get_embedding(query)

    # Thực hiện tìm kiếm trên index FAISS:
    # - Đầu vào: vector của query (1 hàng, n cột)
    # - Trả về: khoảng cách (distances) và chỉ số (indices) của các vector gần nhất
    distances, indices = index.search(np.array([query_vec]), top_k)

    # Lấy ra các metadata tương ứng với top_k chỉ số tìm được
    results = [metadata[i] for i in indices[0]]

    # Trả về danh sách kết quả (ví dụ: các đoạn văn bản hoặc tài liệu tương ứng)
    return results
