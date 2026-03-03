import gradio as gr
import os
from document import load_document, split_documents
from vector_stores import build_vectorstore
from build_rag import chat_interface, summarize_contract

# --- SHARED STATE ---
# simple list to hold docs so it's accessible globally in the script
state = {
    "docs": None,
    "vector_db": None,
    "last_question": None,
    "last_answer": None
}

def process_file(file):
    if file is None:
        return "No file uploaded."

    try:
        state["docs"] = load_document(file.name)
        print("DOCS:", state["docs"])
        chunks = split_documents(state["docs"])
        print("CHUNKS:", len(chunks))
        state["vector_db"] = build_vectorstore(chunks)
        return f"Successfully indexed {len(state['docs'])} pages!"
    except Exception as e:
        return f"ERROR: {str(e)}"


def chat_func(message, history):
    try:
      return chat_interface(message, state["vector_db"])
    except Exception as e:
      return f"Error: {str(e)}"


# SUMMARY TAB FUNCTION
def summarize_document():
    if state["docs"] is None:
        return "Please upload a document first."

    try:
        summary = summarize_contract(state["docs"])
        return summary
    except Exception as e:
        return f"Error summarizing: {str(e)}"



# --- UI DEFINITION ---
# Gradio 6.0
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Smart Document RAG Assistant")

    # -------- TAB 1 --------
    with gr.Tab("1️⃣ Ingestion"):
        file_input = gr.File(label="Upload PDF/DOCX")
        upload_btn = gr.Button("Process Document")
        status = gr.Textbox(label="Status")
        upload_btn.click(process_file, inputs=file_input, outputs=status)

    # -------- TAB 2 --------
    with gr.Tab("2️⃣ Chat & Q&A"):
        gr.ChatInterface(fn=chat_func)

    # -------- TAB 3 --------
    with gr.Tab("3️⃣ Document Summary"):
        summary_btn = gr.Button("Generate Summary")
        summary_output = gr.Textbox(label="Summary", lines=10)
        summary_btn.click(summarize_document, outputs=summary_output)


if __name__ == "__main__":
    demo.launch(share=True, debug=True)






