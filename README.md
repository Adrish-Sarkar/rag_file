# rag_file
# Secure Document Q&A Bot

A Retrieval-Augmented Generation (RAG) application built with LangChain, Gradio, and TinyLlama. It allows users to upload PDF documents and ask questions based strictly on the uploaded content, protected by an input safety guardrail.

## Features

* PDF Upload and Ingestion: Extracts and chunks text from uploaded PDF files.
* Vector Search: Utilizes FAISS and sentence-transformer embeddings for accurate context retrieval.
* Input Guardrails: Screens user queries against prompt injections and restricted keywords before processing.
* Local Inference: Uses TinyLlama-1.1B-Chat for resource-efficient text generation.

## Project Structure

* app.py: The core application containing the UI, guardrails, and RAG logic.
* requirements.txt: Python dependencies required to run the application.

## Local Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

## Deployment

This repository is structured for direct deployment to Hugging Face Spaces using the Gradio SDK. Ensure both app.py and requirements.txt are in the root directory.
