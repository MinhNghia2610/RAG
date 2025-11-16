# generator.py
# Tạo câu trả lời từ query + context (RAG).
# Nếu có OPENAI và USE_OPENAI_FOR_GENERATION=True thì sẽ gọi OpenAI ChatCompletion (tuỳ chọn).
# Nếu không, fallback dùng một chiến lược local: tìm câu/đoạn chứa nhiều token chung (keyword matching)
# rồi ghép lại và trả lời ngắn gọn.

from typing import List, Dict
from config import USE_OPENAI_FOR_GENERATION, OPENAI_API_KEY, OPENAI_MODEL
import os
from utils import normalize_text
import re

def _simple_keyword_answer(query: str, docs: List[Dict], max_sentences: int = 6) -> str:
    """
    Chiến lược nhẹ:
    - tách query thành các từ quan trọng (loại bỏ stopwords đơn giản)
    - duyệt qua từng doc, lấy câu chứa nhiều từ khớp nhất
    - ghép thành context và trả về lời ngắn gọn
    """
    q = normalize_text(query).lower()
    tokens = re.findall(r"\w+", q)
    # loại stopwords rất cơ bản (tiếng Việt đơn giản)
    stop = {"là","và","của","với","cho","những","một","một số","các","the","a","an","to","in"}
    keywords = [t for t in tokens if t and t not in stop and len(t)>1]
    scored_sentences = []
    for d in docs:
        text = d.get("text", "")
        # tách câu
        sents = re.split(r'(?<=[.?!])\s+', text)
        for s in sents:
            s_norm = s.lower()
            score = sum(1 for k in keywords if k in s_norm)
            if score>0:
                scored_sentences.append((score, s.strip(), d.get("source","")))
    # sắp xếp theo score giảm dần
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = scored_sentences[:max_sentences]
    # nếu không có gì khớp -> trả về các đoạn top docs
    if not selected:
        # ghép các doc text ngắn gọn
        ctx = "\n\n".join(d.get("text","")[:800] for d in docs[:3])
        return f"Minh họa thông tin liên quan:\n{ctx}\n\nMình chưa tìm thấy đoạn nào khớp trực tiếp với câu hỏi. Bạn có thể hỏi lại rõ hơn (thông tin cần biết: ví dụ thời gian, tên, bước thực hiện...)."

    out = []
    for score, sent, src in selected:
        out.append(f"- ({score}) {sent}  [{src}]")
    # tạo phần trả lời tóm tắt (rất đơn giản)
    summary = " ".join(s for _, s, _ in selected[:3])
    answer = f"Dựa trên các tài liệu mình tìm được, những đoạn liên quan:\n" + "\n".join(out) + "\n\nTóm tắt trả lời: " + summary
    return answer

def generate_answer(query: str, docs: List[Dict], max_output_tokens: int = 512) -> str:
    """
    Trả về string là câu trả lời cho user.
    """
    # nếu cấu hình dùng OpenAI
    if USE_OPENAI_FOR_GENERATION and OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            context = "\n\n".join(f"Source: {d.get('source')}\nContent: {d.get('text')}" for d in docs[:6])
            prompt = (
                f"You are an assistant for students. Use only the information in CONTEXT to answer the QUESTION. "
                f"If context doesn't contain the answer, say you don't know and suggest how to ask better.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer concisely in Vietnamese."
            )
            # gọi ChatCompletion tạo câu trả lời
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":"You are a helpful assistant for students."},
                          {"role":"user","content":prompt}],
                max_tokens=max_output_tokens,
                temperature=0.0,
            )
            answer = resp["choices"][0]["message"]["content"].strip()
            return answer
        except Exception as e:
            # nếu gọi OpenAI lỗi, fallback
            print("[WARN] Lỗi khi gọi OpenAI. Chuyển sang chế độ local. Lỗi:", e)

    # fallback local: simple keyword matching
    return _simple_keyword_answer(query, docs)
