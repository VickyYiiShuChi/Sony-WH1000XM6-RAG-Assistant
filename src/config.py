import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Use the writable runtime directory on Streamlit Cloud when configured.
RUNTIME_DIR = Path(
    os.getenv("SONY_RAG_RUNTIME_DIR", str(BASE_DIR))
)

DATA_DIR = RUNTIME_DIR / "data"
CHROMA_DIR = RUNTIME_DIR / "chroma_db"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

# Chroma
COLLECTION_NAME = "sony_manual"

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 4

# Generation
MAX_NEW_TOKENS = 400

# Create required directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
