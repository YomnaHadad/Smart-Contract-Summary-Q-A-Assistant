# Smart-Contract-Summary-Q-A-Assistant
**end-to-end Retrieval-Augmented Generation (RAG) web application system** allows users to upload long documents (contracts, insurance policies, reports) and interact with them via a conversational assistant. Users upload PDF/DOCX, the system extracts, chunks, and embeds content, stores it in a vector store, and enables chat-based question answering with guard-rails and source citations.

## 🏗️ System Architecture
The project is structured into modular components:

### 1️⃣ Document Ingestion

* Supports **PDF** (via PyMuPDF) and **DOCX** (via Docx loader).
* Documents are split into manageable text chunks using `RecursiveCharacterTextSplitter`.
* Chunk size: 1000 characters with overlap to preserve context.

### 2️⃣ Vector Storage & Retrieval

* Text chunks are embedded using **HuggingFace `all-MiniLM-L6-v2`**.
* Embeddings are stored in a **FAISS vector database**.
* A retriever fetches semantically relevant chunks based on user queries.

### 3️⃣ Question Answering (RAG Pipeline)

* Retrieved chunks are formatted with metadata (source + page).
* A prompt template ensures answers:

  * Use **only retrieved context**
  * Include citation information if available
* The LLM (Groq-hosted model) generates the final response.

### 4️⃣ Document Summarization

* A separate summarization chain uses the first document chunks.
* Generates a concise structured summary.
* Designed originally for contracts but works for general documents.

## 🖥️ User Interface (Gradio)

The interface contains four interactive tabs:

1. **Ingestion** – Upload and index document.
2. **Chat & Q&A** – Ask contextual questions.
3. **Document Summary** – Generate overall document summary.


## **Objectives**
- Demonstrate understanding of LLM inference interfaces and microservices.
- Build an end-to-end RAG (Retrieval Augmented Generation) pipeline.
- Showcase strategies for long-form document processing.
- Apply embeddings and semantic similarity for guardrailing.
- Evaluate and validate retrieval/answer quality.

## **Scope**
- File ingestion: PDF/DOCX.
- Chunking & embedding.
- Vector store setup (Chroma or FAISS).
- Retrieval + LLM answer generation.
- Chat interface with history.
- Guard-rails for safety and factuality.
- Local deployment with FastAPI/LangServe.

## **Technology Stack**
- LangChain, LangServe, FastAPI
- Gradio (UI)
- FAISS (vector store)
- OpenAI embeddings
- PyMuPDF, pdfplumber, python-docx (file parsing)
