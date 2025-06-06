# Build_fast_ai_intern
# 🧠 Contextual Medical Q&A Chatbot with RAG Pipeline
## WEBAPP LINK -: https://medical-ques-ans.streamlit.app/

> **External Sources Used**:  
> ChatGPT (OpenAI) for design guidance and structuring,  
> [LangChain Documentation](https://docs.langchain.com),  
> [Gemini API Documentation](https://ai.google.dev)

This project demonstrates a **domain-specific retrieval-augmented generation (RAG)** pipeline that answers complex medical queries using Gemini Pro and a large offline medical corpus. Built as a demo for GenAI-powered assistant systems with real-world applicability in healthcare and knowledge support.

## 🚀 Project Overview

- Collected a medical corpus (700+ pages), cleaned and chunked into **10,000+ indexed segments**
- Created a **semantic search engine** using FAISS for fast, dense vector retrieval
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

## 📁 Key Files & Structure

- `data_load_and_upload.py` – Loads and prepares medical documents; chunking & vector indexing using FAISS.
- `medi_qabot.py` – Main app logic for the medical Q&A bot; ties retrieval and generation.
- `memory_connect_with_llm.py` – Handles LLM memory/context handling with Gemini API and LangChain.
- `requirements.txt` – Python dependencies for full environment setup.
- `ASKED QUES AND GENERATED ANS.pdf` – Sample outputs demonstrating system performance.
- `vectorstore/db_faiss/` – Stored FAISS index for semantic retrieval.
- `data/` – Raw or preprocessed medical documents.

## 🏆 Highlights

- Efficient handling of **domain-specific**, multi-page unstructured data
- Fully local FAISS-based retrieval — no external dependency
- RAG architecture demonstrates **low-latency, context-aware response generation**
- Ideal for use cases in **healthcare, education, and sales assistant AI**

## 🧪 Sample Questions

- “What is the difference between acute and chronic hepatitis?”
- “When should a patient with cirrhosis be hospitalized?”
- “Explain the function of the liver in one paragraph.”

---

📌 Built to simulate real-world RAG use cases in medical QA, this project mirrors the architecture behind scalable, multilingual GenAI copilots like Darwix’s assist engine.
