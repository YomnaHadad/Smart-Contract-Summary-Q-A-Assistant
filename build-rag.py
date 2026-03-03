from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable import RunnableParallel
from vector_stores import build_vectorstore
from langchain.llms import OpenAI
from document import load_document, split_documents


## Reorders longer documents to center of output text
from langchain.document_transformers import LongContextReorder
long_reorder = RunnableLambda(LongContextReorder().transform_documents)

# API
from google.colab import userdata
openai_api_key = userdata.get('OPENAI_API_KEY')
instruct_llm = OpenAI(openai_api_key=openai_api_key, model_name= "gpt-4o-mini")

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
    # This helper combines the text content with its source info
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[Source: {source}, Page: {page}]\nContent: {doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def process_file(file):
    global vector_db
    # file.name is the path to the uploaded PDF/DOCX
    docs = load_document(file.name)
    chunks = split_documents(docs)
    vector_db = build_vectorstore(chunks)
    return "Document processed successfully! You can now ask questions."


def chat_interface(question, history):
    if vector_db is None:
        return "Please upload a document first."
    
    retriever = vector_db.as_retriever() 
    rag_chain = (
      RunnableParallel({
        "context": retriever.as_retriever() | long_reorder | RunnableLambda(format_docs_with_metadata),
        "question": RunnablePassthrough()
      })
      | prompt | instruct_llm | StrOutputParser()
    )
    response = rag_chain.invoke(question)
    return response


def summarize_contract(docs):
    summary_prompt = ChatPromptTemplate.from_template(
        "Write a concise summary of the following legal contract. "
        "Highlight key parties, obligations, and termination clauses."
        "\n\nContract Content:\n{text}"
    )
    
    # "Stuffing" the first few chunks for a quick summary
    text_content = "\n".join([d.page_content for d in docs[:5]]) 
    
    chain = summary_prompt | instruct_llm | StrOutputParser()
    return chain.invoke({"text": text_content})   
