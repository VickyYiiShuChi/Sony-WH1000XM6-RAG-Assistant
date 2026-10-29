
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.collection = None

    def create_collection(self, reset=False):
        if reset:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        try:
            self.collection = self.client.get_collection(
                COLLECTION_NAME
            )
        except Exception:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        return self.collection

    def embed(self, texts):
        vectors = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def add_chunks(self, chunks, batch_size=64):
        if self.collection is None:
            self.create_collection()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            texts = [item["text"] for item in batch]
            embeddings = self.embed(texts)

            ids = [
                f"chunk_{start + i}"
                for i in range(len(batch))
            ]

            metadatas = [
                {
                    "source": item.get("source", "unknown"),
                    "chapter": item["chapter"][:500],
                    "section": item["section"][:500],
                    "page_start": int(item["page_start"]),
                    "page_end": int(item["page_end"]),
                    "section_index": int(item["section_index"]),
                    "chunk_index": int(item["chunk_index"]),
                }
                for item in batch
            ]

            # Upsert supports repeatable IDs if the same batch is indexed.
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        return self.collection.count()

    def search(self, question, top_k=4, chapter=None):
        if self.collection is None:
            self.create_collection()

        if self.collection.count() == 0:
            return []

        query_embedding = self.embed([question])[0]

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if chapter:
            kwargs["where"] = {"chapter": chapter}

        result = self.collection.query(**kwargs)

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            {
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
            }
            for text, metadata, distance
            in zip(documents, metadatas, distances)
        ]
