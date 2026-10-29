# AI Knowledge Assistant: Structure-Aware RAG

A Retrieval-Augmented Generation (RAG) application that answers questions about the Sony WH-1000XM6 headphones using information retrieved from the official user manual.

Built with Python, Sentence Transformers, ChromaDB, Groq, and Streamlit, this project explores how document structure and semantic retrieval can help an LLM generate answers grounded in a specific knowledge source.

## Live Demo

**Try the application online:** [Sony RAG Assistant](YOUR_STREAMLIT_APP_URL)

No local installation is required. Open the web app, enter a question about the Sony WH-1000XM6 headphones, and review the generated answer and its source references.

> The first request may take longer if the application needs to initialize its models or build the document index.

## Screenshots

The following screenshots demonstrate the application's interface and example outputs.

<!-- Replace these paths with screenshots committed to the repository. -->

### Chat Interface

![Sony RAG Assistant chat interface](screenshots/chat-interface.png)

The Streamlit interface allows users to ask questions about the headphones in natural language.

### Example: Answer with Source References

![Example RAG answer with source references](screenshots/answer-with-sources.png)

The system retrieves relevant passages from the user manual and generates an answer with source metadata, such as section information and PDF page references, when available.

### Retrieval and Document Processing

![Document processing and retrieval example](screenshots/retrieval-example.png)

This example illustrates how document content is processed into chunks and retrieved based on semantic similarity.

*Screenshots should show actual outputs from the application. Remove this section's image placeholders if the corresponding screenshots have not yet been added.*

## Why I Built This Project

General-purpose AI assistants can answer a wide range of questions, so building a manual-specific assistant is not intended to outperform them.

Instead, this project is a practical learning exercise to understand how a RAG system works from end to end.

The main objectives are to:

- Understand how to extract and clean information from PDF documents.
- Explore how document headings and section boundaries can improve text chunking.
- Learn how semantic embeddings and vector databases support information retrieval.
- Connect retrieved document context to an LLM for grounded answer generation.
- Provide source references so users can verify answers against the original document.
- Deploy an AI application as a usable web demo.

The project also explores an important limitation of LLM applications: a fluent answer is not necessarily a correct answer. Retrieval quality, document processing, and source verification all matter.

## Features

- **PDF text extraction:** Extracts text from the user manual using PyMuPDF.
- **Document cleaning:** Processes extracted text to reduce formatting noise.
- **Structure detection:** Identifies document sections and headings using configurable heuristics.
- **Section-aware chunking:** Splits content into chunks while considering document structure.
- **Semantic embeddings:** Uses Sentence Transformers with `all-MiniLM-L6-v2` to represent text as vectors.
- **Vector storage:** Stores document chunks and metadata in ChromaDB.
- **Semantic retrieval:** Retrieves relevant passages for a user's question.
- **LLM answer generation:** Uses the Groq API to generate answers from retrieved context.
- **Source references:** Includes available chapter, section, and PDF page metadata.
- **Interactive web interface:** Provides a Streamlit chat interface.
- **Notebook demonstration:** Includes a Jupyter Notebook for inspecting the RAG pipeline.

## Architecture

```text
Sony WH-1000XM6 User Manual (PDF)
                |
                v
       PDF Text Extraction
                |
                v
        Document Cleaning
                |
                v
      Structure Detection
                |
                v
      Section-Aware Chunking
                |
                v
     Sentence Transformer
      (all-MiniLM-L6-v2)
                |
                v
          ChromaDB
        Vector Store
                |
                v
        Semantic Retrieval
                |
                v
       Retrieved Context
                |
                v
       Groq LLM Generation
                |
                v
       Answer with Sources
```

The pipeline separates document preparation from question answering. The manual is processed into chunks and indexed, then relevant chunks are retrieved when a user submits a question.

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Language | Python | Application and pipeline logic |
| Web interface | Streamlit | Interactive chat interface |
| PDF processing | PyMuPDF | Text extraction |
| Embeddings | Sentence Transformers | Semantic text representation |
| Embedding model | `all-MiniLM-L6-v2` | Text embeddings |
| Vector database | ChromaDB | Store and retrieve document chunks |
| LLM API | Groq | Answer generation |
| LLM model | `openai/gpt-oss-20b` | Language generation, as configured in the application |

## Project Structure

```text
sony-rag-assistant/
├── app.py
├── requirements.txt
├── README.md
├── demo.ipynb
├── data/
│   └── Sony_WH1000XM6_UserManual.pdf
└── src/
    ├── config.py
    ├── document_parser.py
    ├── document_cleaner.py
    ├── structure_extractor.py
    ├── chunker.py
    ├── vector_store.py
    ├── retriever.py
    ├── generator.py
    └── rag_pipeline.py
```

The `data/` directory contains the source document, while `src/` contains the document processing, indexing, retrieval, and generation components.

The vector database may be created at runtime, depending on the application's configuration.

## Try the Online Demo

1. Open the [Live Demo](YOUR_STREAMLIT_APP_URL).
2. Wait for the application to finish initializing.
3. Enter a question about the Sony WH-1000XM6 headphones.
4. Review the answer and its source references.
5. Ask a follow-up question to explore another feature.

Example questions:

- How do I turn on noise cancelling?
- How can I connect the headphones to two devices?
- How do I charge the headphones?
- What should I do if the headphones do not connect?
- Can I use these headphones underwater?

The final question is useful for checking whether the system handles questions that may not be supported by the manual. Its response should be checked against the retrieved evidence rather than assumed to be correct.

## Run Locally

### Prerequisites

- Python 3.10 or another Python version supported by the dependencies.
- Git.
- A Groq API key.
- The Sony WH-1000XM6 user manual, subject to applicable distribution permissions.

### 1. Clone the Repository

```bash
git clone https://github.com/VickyYiiShuChi/sony-rag-assistant.git
cd sony-rag-assistant
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API Key

Set your API key as an environment variable.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

macOS or Linux:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

Replace the placeholder with your own key. Do not commit API keys to GitHub.

### 5. Prepare the Source Document

Ensure that the manual is available at the path expected by the application:

```text
data/Sony_WH1000XM6_UserManual.pdf
```

If you cannot redistribute the PDF, obtain it separately from an authorized source and place it in the expected location. Do not publish copyrighted documents without permission.

### 6. Launch the Application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

The application must be able to access the source document and initialize or access its vector index. First-time startup may take longer while models are downloaded or the index is built.

## Limitations

- **Heuristic structure detection:** Headings and section boundaries may not always be detected correctly.
- **PDF format dependency:** Scanned or image-based documents may require OCR.
- **Retrieval limitations:** Relevant information may not always be retrieved, especially for vague or complex questions.
- **LLM limitations:** Generated answers may contain errors or unsupported statements.
- **Source verification:** Users should verify important information against the original manual.
- **PDF page numbering:** Referenced PDF page positions may differ from printed page numbers.
- **Deployment constraints:** Initial loading time and availability depend on the hosting environment, model initialization, and external API services.

## Future Improvements

- Evaluate retrieval quality using a set of representative questions and expected source passages.
- Add automated tests for document processing, retrieval, and answer generation.
- Improve retrieval with metadata filtering and configurable retrieval parameters.
- Add clearer source previews and links to relevant manual pages.
- Improve caching and startup performance.
- Support additional documents and configurable knowledge sources.

## Learning Outcomes

This project provided practical experience with document processing, semantic embeddings, vector databases, retrieval-augmented generation, LLM API integration, and deployment of an interactive AI application.

The main takeaway is that building a useful RAG system involves more than connecting an LLM to a vector database. Document quality, chunking strategy, retrieval relevance, context construction, and answer verification all influence the final result.

## Disclaimer

This is an independent educational project and is not affiliated with or endorsed by Sony. The application is intended for demonstration and learning purposes. Consult the original user manual for authoritative product instructions.
