# interface.py
# Giao diện CLI đơn giản để thử chatbot RAG

from retriever import get_default_retriever
from generator import generate_answer
from config import DEFAULT_TOP_K
import textwrap

def run_cli():
    print("="*60)
    print("RAG Chatbot (CLI) - Nhập 'exit' để thoát, 'help' để xem lệnh.")
    print("="*60)
    retriever = get_default_retriever()
    while True:
        try:
            q = input("\nBạn hỏi> ").strip()
        except KeyboardInterrupt:
            print("\nThoát.")
            break
        if not q:
            continue
        if q.lower() in ("exit","quit"):
            print("Tạm biệt!")
            break
        if q.lower() in ("help","h"):
            print("Gõ câu hỏi bình thường. Gõ 'exit' để thoát.")
            continue

        # lấy top docs
        docs = retriever.retrieve(q, top_k=DEFAULT_TOP_K)
        if not docs:
            print("Không tìm thấy dữ liệu. Hãy chắc bạn đã build embeddings hoặc có file trong data/.")
            continue

        # in ra context tóm tắt
        print("\n--- Tài liệu liên quan (top) ---")
        for i, d in enumerate(docs, 1):
            print(f"[{i}] score={d['score']:.4f} source={d['source']}")
            snippet = d['text'][:400].replace("\n"," ")
            print(textwrap.fill(snippet, width=100))
            print("-"*40)

        # tạo câu trả lời
        answer = generate_answer(q, docs)
        print("\n=== Trả lời ===")
        print(answer)
        print("="*60)

if __name__ == "__main__":
    run_cli()
