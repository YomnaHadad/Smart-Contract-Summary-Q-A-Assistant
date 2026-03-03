# Smart-Contract-Summary-Q-A-Assistant
web application that allows users to upload long documents (contracts, insurance policies, reports) and interact with them via a conversational assistant. Users upload PDF/DOCX, the system extracts, chunks, and embeds content, stores it in a vector store, and enables chat-based question answering with guard-rails and source citations.

2. **Objectives**
- Demonstrate understanding of LLM inference interfaces and microservices.
- Build an end-to-end RAG (Retrieval Augmented Generation) pipeline.
- Showcase strategies for long-form document processing.
- Apply embeddings and semantic similarity for guardrailing.
- Evaluate and validate retrieval/answer quality.

3. **Scope**
- File ingestion: PDF/DOCX.
- Chunking & embedding.
- Vector store setup (Chroma or FAISS).
- Retrieval + LLM answer generation.
- Chat interface with history.
- Guard-rails for safety and factuality.
- Local deployment with FastAPI/LangServe.

4.**Technology Stack**
- LangChain, LangServe, FastAPI
- Gradio (UI)
- FAISS (vector store)
- OpenAI embeddings
- PyMuPDF, pdfplumber, python-docx (file parsing)
