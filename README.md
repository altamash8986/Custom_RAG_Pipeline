# 📄 Custom Retrieval-Augmented Generation (RAG) Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-all--MiniLM--L6--v2-yellow)

A lightweight, fully custom-built Retrieval-Augmented Generation (RAG) backend. 

Unlike standard tutorials that hide the mechanics behind heavy wrappers (like LangChain's `RetrievalQA` chains), this project implements the core components from scratch. It handles document ingestion, semantic chunking, local vectorization, and mathematical similarity search through custom object-oriented Python classes.

## 🏗️ System Architecture

The pipeline consists of four distinct phases:

1. **Ingestion (`load_all_pdfs`)**: Scans the active directory for `.pdf` files and parses the text data and document metadata.
2. **Chunking (`split_document`)**: Uses a `RecursiveCharacterTextSplitter` to break large documents into 500-character chunks with a 50-character overlap. This overlap is crucial for preventing context loss when a sentence crosses a chunk boundary.
3. **Embedding (`EmbeddingsManager`)**: Passes the chunks through Hugging Face's `all-MiniLM-L6-v2` transformer model. This converts raw text into dense, 384-dimensional mathematical vectors representing the semantic meaning of the text.
4. **Storage & Retrieval (`VectorManager` & `RAG_Retriever`)**: 
   - Stores the vectors and metadata in a persistent local **ChromaDB** instance.
   - When a user submits a query, it is embedded using the same model.
   - The engine calculates the spatial distance between the query vector and the document vectors, converting the distance into a **Similarity Score** and returning the highest-ranking contexts.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Document Processing:** LangChain (`PyPDFLoader`, `RecursiveCharacterTextSplitter`)
* **Machine Learning / NLP:** `sentence-transformers` (Hugging Face)
* **Vector Database:** ChromaDB

---

## 🚀 Installation & Setup

You can run this engine natively on your local machine or via a Docker container.

### Option 1: Local Environment

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/altamash8986/Custom_RAG_Pipeline.git](https://github.com/altamash8986/Custom_RAG_Pipeline.git)
   cd Custom_RAG_Pipeline

# 👨‍💻 Author

## Mohd Altamash

B.Tech Computer Science Engineering

AI and ML Engineer

## 🌐 Connect With Me

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/mohd-altamash-0997592a6/)
- [![LeetCode](https://img.shields.io/badge/LeetCode-Profile-orange?logo=leetcode)](https://leetcode.com/u/altamash007/)

---


