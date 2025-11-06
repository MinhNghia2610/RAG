from openai import OpenAI
from src.config import *

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(contexts, query):
    context_text = "\n".join([c["text"] for c in contexts])
    prompt = f"""
    Dưới đây là nội dung tham khảo:
    {context_text}

    Câu hỏi: {query}

    → Hãy trả lời ngắn gọn, tự nhiên và chính xác nhất.
    """

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()
