# RAG-pdf-chatbot
PDF Q&amp;A Chatbot with RAG – A  Retrieval-Augmented Generation pipeline for querying PDF documents. Extracts text, generates embeddings and answers questions using FLAN-T5. Experimental RAG pipeline for PDF question answering. Uses LangChain, HuggingFace embeddings, and FLAN-T5 for retrieval-augmented generation. Currently testing chunking strategies and prompt engineering to reduce hallucinations.  

[![GitHub Stars](https://img.shields.io/github/stars/anuraga-sankepally/RAG-pdf-chatbot?style=social)](https://github.com/anuraga-sankepally/RAG-pdf-chatbot)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Demo GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcW0yY2VtY2F4eGJmN3B6dWJ6Y2J4b2VlZ3B6eHl1dGZ2a3B6eCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT5LMHxhOfscxPfIfm/giphy.gif)  

## 🚀 Features

- **PDF Text Extraction** with smart chunking
- **Semantic Search** using FAISS vectorstore
- **LLM Answers** via FLAN-T5 (or your preferred model)
- **Hybrid Retrieval** combining keywords + vectors
- **Secret Scanning Protected** - No accidental API leaks

## ⚡ Quick Start

```bash
# Clone the repo
git clone https://github.com/anuraga-sankepally/RAG-pdf-chatbot.git
cd RAG-pdf-chatbot

# Install dependencies
pip install -r requirements.txt

# Add your PDF to data/ and run!
python app/rag_pipeline.py
