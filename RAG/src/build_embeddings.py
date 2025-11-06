import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n")
    return text

def build_vector_store():
    urls = [
        "https://openai.com/research",
        "https://platform.openai.com/docs/guides/retrieval",
        "https://en.wikipedia.org/wiki/OpenAI"
    ]

    documents = []
    for url in urls:
        print(f"📥 Fetching {url} ...")
        text = get_text_from_url(url)
        documents.append(text)

    if not documents:
        print("⚠️ Không có nội dung nào được tải về.")
        return

    print("✅ Fetched", len(documents), "web pages.")

    texts = []
    for doc in documents:
        # Cắt nhỏ nội dung để embedding dễ hơn
        chunks = [doc[i:i+1000] for i in range(0, len(doc), 1000)]
        texts.extend(chunks)

    print("Creating embeddings...")

    vectors = []
    for chunk in texts:
        emb = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        vectors.append(emb.data[0].embedding)

    vectors = np.array(vectors)
    print("✅ Embeddings created successfully with shape:", vectors.shape)

if __name__ == "__main__":
    build_vector_store()
