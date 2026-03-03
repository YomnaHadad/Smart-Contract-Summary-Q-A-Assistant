from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def build_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
