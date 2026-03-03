# - File ingestion: PDF/DOCX.

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
import os

#_____________________________________

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    docs_split = text_splitter.split_documents(documents)
    return docs_split
#_____________________________________

def load_document(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyMuPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use PDF or DOCX.")

    docs = loader.load()
    print(f"Loaded {len(docs)} page(s) from {os.path.basename(file_path)}")
    return docs

