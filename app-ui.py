import gradio as gr
from build_rag import load_document, split_documents, build_vectorstore, context_prompt, instruct_llm
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from build-rag import process_file, chat_interface, summarize_contract

# 1. Global variables to hold our data across different button clicks
vector_db = None
loaded_docs = None 

def process_file(file):
    global vector_db, loaded_docs
    if file is None:
        return "No file uploaded."
    
    loaded_docs = load_document(file.name) 
    chunks = split_documents(loaded_docs)
    vector_db = build_vectorstore(chunks)
    
    return f"Successfully processed {len(loaded_docs)} pages. Now you can summarize or chat!"

def handle_summarize():
    global loaded_docs
    if loaded_docs is None:
        return "Please upload and process a document first."
    
    return summarize_contract(loaded_docs)

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📜 Smart Contract Assistant")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF or DOCX")
            upload_btn = gr.Button("1. Process & Index")
            summary_btn = gr.Button("2. Generate Summary")
            summary_out = gr.Textbox(label="Contract Summary", lines=10)
            
        with gr.Column(scale=2):
            gr.ChatInterface(fn=chat_interface, title="Chat with your Contract")

    # 2. Corrected Event Handlers
    # Notice we don't pass 'docs' here; the function finds 'loaded_docs' globally
    upload_btn.click(fn=process_file, inputs=file_input, outputs=summary_out)
    summary_btn.click(fn=handle_summarize, inputs=None, outputs=summary_out)

demo.launch(share=True, debug=True)
    

