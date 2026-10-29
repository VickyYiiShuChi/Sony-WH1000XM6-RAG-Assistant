# AI Knowledge Assistant: Structure-Aware RAG

A Retrieval-Augmented Generation (RAG) application that answers
questions about the Sony WH-1000XM6 user manual using document retrieval
and a large language model powered by the Groq API.

## Features

- PDF text extraction using PyMuPDF
- Document cleaning and section structure detection
- Section-aware text chunking
- Semantic embeddings using Sentence Transformers
- Vector storage and retrieval using ChromaDB
- Answer generation through the Groq API
- Source metadata including chapter, section, and PDF pages
- Interactive Streamlit web interface
- Jupyter Notebook demonstration of the RAG pipeline

## Architecture

PDF Manual
-> Text Extraction
-> Document Cleaning
-> Structure Detection
-> Section-Aware Chunking
-> MiniLM Embeddings
-> ChromaDB Vector Store
-> Semantic Retrieval
-> Groq LLM Answer Generation
-> Answer with PDF Page References

## Tech Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- ChromaDB
- Groq API
- Hugging Face

## Project Structure

    app.py
    requirements.txt
    README.md
    demo.ipynb
    data/
    src/
        config.py
        document_parser.py
        document_cleaner.py
        structure_extractor.py
        chunker.py
        vector_store.py
        retriever.py
        generator.py
        rag_pipeline.py

## Setup

1. Clone this repository.
2. Install the dependencies:

       pip install -r requirements.txt

3. Configure your Groq API key using an environment variable:

       GROQ_API_KEY=your_api_key

   Do not commit API keys or other secrets to version control.

4. Ensure the required document and vector index are available before
   running the application.

## Run the Web Application

    streamlit run app.py

Configure `GROQ_API_KEY` in your environment before starting the app.

## Run the Notebook

Open `demo.ipynb` in Jupyter Notebook or Google Colab and run the
cells in order to inspect document parsing, chunking, vector indexing,
retrieval, and answer generation.

## Limitations

- Heading and section detection use configurable heuristics.
- Scanned PDFs may require OCR.
- Answer quality depends on retrieval relevance and the available
  document context.
- Generated answers should be checked against the cited source pages.
- PDF page references identify PDF page positions, which may differ
  from printed page numbers in the manual.
