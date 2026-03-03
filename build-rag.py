import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel
from vector_stores import build_vectorstore
from langchain_groq import ChatGroq
from langchain.llms import OpenAI
from document import load_document, split_documents


## Reorders longer documents to center of output text
# from langchain.document_transformers import LongContextReorder
# long_reorder = RunnableLambda(LongContextReorder().transform_documents)

# API
groq_api_key =  os.environ["GROQ_API_KEY"] 
instruct_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key= groq_api_key,
    temperature=0
)

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "Include the source/page number in your answer if available."
    "\n\nContext: {context}"
    "\n\nQuestion: {question}"
)

# Global variable to hold our vectorstore once a file is uploaded
vector_db = None


def format_docs_with_metadata(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page: {page}]\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def process_file(file_path):
    global vector_db
    try:
        # Load the document (PDF or DOCX)
        docs = load_document(file_path)
        if not docs:
            return "Error: No pages found in the document."

        # Split into chunks for vector store
        chunks = split_documents(docs)
        if not chunks:
            return "Error: Failed to split document into chunks."

        # Build vectorstore
        vector_db = build_vectorstore(chunks)
        return f"Document processed successfully! {len(chunks)} chunks indexed."
    except Exception as e:
        return f"Error processing document: {str(e)}"


def chat_interface(question, vector_db_local):
    if vector_db_local is None:
        return "Please upload a document first."

    try:
        retriever = vector_db_local.as_retriever()
        rag_chain = (
            RunnableParallel({
                "context": retriever | RunnableLambda(format_docs_with_metadata),
                "question": RunnablePassthrough()
            })
            | prompt | instruct_llm | StrOutputParser()
        )
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"Error in chat: {str(e)}"


def summarize_contract(docs):
    try:
        summary_prompt = ChatPromptTemplate.from_template(
            "Write a concise summary of the following legal contract. "
            "Highlight key parties, obligations, and termination clauses.\n\n"
            "Contract Content:\n{text}"
        )

        text_content = "\n".join([d.page_content for d in docs[:5]])  # first 5 chunks
        chain = summary_prompt | instruct_llm | StrOutputParser()
        return chain.invoke({"text": text_content})
    except Exception as e:
        return f"Error summarizing contract: {str(e)}"  
