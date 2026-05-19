# 🧠 RAG Chatbot — Retrieval-Augmented Generation System

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with FAISS vector search, sentence-transformers embeddings, and flexible LLM backend support.

**Live architecture:** Document retrieval → Semantic embedding → Vector search → LLM generation

---

## ✨ Features

- **🔍 Semantic Search** — FAISS vector store with `all-MiniLM-L6-v2` embeddings for efficient document retrieval
- **🤖 Dual LLM Backend** — Supports OpenAI API (GPT-4/GPT-3.5) with local model fallback for offline/air-gapped scenarios
- **⚙️ Configurable Pipeline** — Adjustable chunking strategies, embedding models, retriever top-k, and prompt templates
- **🧩 Modular Architecture** — Clean separation of retriever, generator, and configuration modules for maintainability

---

## 🏗️ Architecture

```
RAG/
├── src/
│   ├── config.py        # Environment & model configuration
│   ├── retriever.py     # Document loading, chunking, embedding, FAISS indexing
│   ├── generator.py     # LLM query generation (OpenAI + local fallback)
├── data/                # Document storage
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variables template
```

### Processing Flow

```
User Query
    ↓
retriever.py — Encode query → FAISS vector search → Retrieve top-K chunks
    ↓
generator.py — Augmented prompt → LLM (OpenAI/local) → Generated response
    ↓
Context-aware answer returned to user
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key (optional — for cloud inference)

### Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_api_key_here"
```

### Usage

```python
from src.retriever import Retriever
from src.generator import Generator

retriever = Retriever()
generator = Generator()

query = "What are the key concepts in this document?"
chunks = retriever.search(query, top_k=5)
response = generator.generate(query, chunks)
print(response)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Store | **FAISS** (Facebook AI Similarity Search) |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) |
| Cloud LLM | **OpenAI API** (GPT-4 / GPT-3.5-turbo) |
| Local LLM | Configurable local model fallback |
| Language | **Python** |

---

## 📈 Use Cases

- **Customer Support Q&A** — Index product documentation for instant answers
- **Research Assistants** — Query academic papers and technical documents
- **Knowledge Bases** — Build organization-specific RAG over internal wikis
- **AI Evaluator Tool** — Test LLM grounding accuracy with retrievable context

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Nghia Le** — [GitHub](https://github.com/MinhNghia2610) • [LinkedIn](https://www.linkedin.com/in/ngh%C4%A9a-l%C3%AA-045377245/)

*Built with AI-native development workflows.*