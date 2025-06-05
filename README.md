# build_fast_ai_intern
# 🧠 Contextual Medical Q&A Chatbot with RAG Pipeline

> **External Sources Used**:  
> ChatGPT (OpenAI) for guidance and code structuring,  
> [LangChain Documentation](https://docs.langchain.com),  
> [Gemini API Documentation](https://ai.google.dev)

This project demonstrates a **domain-specific retrieval-augmented generation (RAG)** pipeline that answers complex medical queries using Gemini Pro and a large offline medical corpus. Built as a demo for GenAI-powered assistant systems with real-world applicability in healthcare and knowledge support.

## 🚀 Project Overview

- Collected a medical corpus (700+ pages), cleaned and chunked into **10,000+ indexed segments**
- Created a **semantic search engine** using `FAISS` for fast, dense vector retrieval
- Built a custom RAG pipeline using:
  - Gemini LLM for generation
  - LangChain for orchestration
  - Streamlit for interactive querying
- End-to-end system returns accurate, context-aware answers to open-ended medical questions

## 🔧 Tech Stack

- `FAISS` (vector store)
- `Gemini API` (LLM inference)
- `LangChain` (pipeline design)
- `LLMIndex (LlamaIndex)` (retrieval)
- `Streamlit` (frontend app)
- `pandas`, `tqdm`, `PyMuPDF` (data processing)

## 🩺 Use Case

Users can ask natural language medical questions like:
> “What are the early symptoms of liver cirrhosis?”

And receive structured answers based on real corpus content — including citations and chunk references.

## 🏆 Highlights

- Efficient handling of **domain-specific**, multi-page unstructured data
- Fully local FAISS-based retrieval — no external dependency
- RAG architecture demonstrates **low-latency, context-aware response generation**
- Ideal for use cases in **healthcare, education, and sales assistant AI**

## 📁 Files

- `app.py`: Streamlit UI for question-answering
- `rag_pipeline.py`: FAISS setup + LLM response logic
- `chunk_utils.py`: Document processing into retrievable units
- `data/`: Includes test corpus

## 🧪 Sample Questions

- “What is the difference between acute and chronic hepatitis?”
- “When should a patient with cirrhosis be hospitalized?”
- “Explain the function of the liver in one paragraph.”

---

📌 Built to simulate real-world RAG use cases in medical QA, this project mirrors the architecture behind scalable, multilingual GenAI copilots like Darwix’s assist engine.

