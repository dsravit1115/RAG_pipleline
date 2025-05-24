# 🔍 Retrieval-Augmented Generation (RAG) - Local Demo

This project is a **local RAG (Retrieval-Augmented Generation)** pipeline using **LangChain**, **FAISS**, and **OpenAI GPT-3.5**.

## 📌 Project Overview

This demo reads a text file, splits it into semantic chunks, embeds them using `sentence-transformers`, stores them in a FAISS vector store, and answers user queries 
by retrieving relevant chunks and generating a response via OpenAI GPT.

### 🔧 Tech Stack

- 🧠 **LLM**: OpenAI GPT-3.5
- 🗂️ **Vector Store**: FAISS (local, in-memory)
- 🔎 **Embeddings**: all-MiniLM-L6-v2 (via SentenceTransformers)
- ⚙️ **Framework**: LangChain
- 📄 **Document Loader**: TextLoader

