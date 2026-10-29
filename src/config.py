from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

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
