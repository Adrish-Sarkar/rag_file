import os
import gradio as gr
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Global variables to track state
retriever = None
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 1. Guardrail System
def input_guardrail(user_query: str) -> tuple[bool, str]:
    banned_keywords = ["hack", "exploit", "bypass", "override", "ignore previous instructions"]
    if len(user_query.strip()) < 3:
        return False, "Your question is too short. Please ask a specific question about the document."
    query_lower = user_query.lower()
    for word in banned_keywords:
        if word in query_lower:
            return False, f"Safety Alert: Your request contains a restricted term ('{word}')."
    return True, user_query

# 2. Ingestion Pipeline
def process_uploaded_file(file):
    global retriever
    if file is None:
        return "Please upload a valid PDF file."
    try:
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 2})
        return "Document successfully indexed! You can now ask questions below."
    except Exception as e:
        return f"Error processing file: {str(e)}"

# 3. Generation Engine
generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

def chat_interface(user_query, history):
    global retriever
    
    # Check if a document is loaded
    if retriever is None:
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": "Please upload and process a PDF document first."})
        return history, ""
        
    # Run Guardrail
    is_safe, message = input_guardrail(user_query)
    if not is_safe:
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": message})
        return history, ""
            
    # Run RAG Retrieval
    docs = retriever.invoke(user_query)
    context = "\n".join([doc.page_content for doc in docs])
        
    # Prompt Generation
    prompt = f"<|system|>\nYou are a helpful assistant. Answer the question based ONLY on the provided context.\nContext:\n{context}<|user|>\n{user_query}<|assistant|>\n"
    outputs = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)
    response = outputs[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
    
    # Correctly append as dictionaries matching the standard message schema
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response})
        
    return history, ""

# 4. Gradio Interface Construction
with gr.Blocks() as demo:
    gr.Markdown("# Secure Document Q&A (RAG Application)")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
            process_btn = gr.Button("Process Document")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Q&A History", type="messages")
            msg = gr.Textbox(label="Ask a question about the document")
            clear = gr.ClearButton([msg, chatbot])
            
        process_btn.click(process_uploaded_file, inputs=[file_input], outputs=[status_output])
    
    # Handle chat updates and clear text box cleanly in a single action
    msg.submit(chat_interface, inputs=[msg, chatbot], outputs=[chatbot, msg])

if __name__ == "__main__":
    demo.launch(share=True)
