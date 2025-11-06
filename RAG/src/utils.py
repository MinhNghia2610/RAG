import os
from PyPDF2 import PdfReader

def load_text_from_file(path):
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        return " ".join(page.extract_text() for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

def split_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
