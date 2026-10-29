
from pathlib import Path

from src.config import TOP_K
from src.document_parser import extract_pdf_pages
from src.document_cleaner import clean_pages
from src.structure_extractor import extract_document_structure
from src.chunker import build_chunks
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import AnswerGenerator


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)
        self.generator = AnswerGenerator()

        self.sections = []
        self.chunks = []
        self.source_path = None

    def process_document(self, pdf_path, remove_toc=False):
        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        pages = extract_pdf_pages(str(pdf_path))
        pages = clean_pages(pages, remove_toc=remove_toc)

        sections = extract_document_structure(pages)
        chunks = build_chunks(sections)

        if not chunks:
            raise ValueError("No chunks were generated.")

        for chunk in chunks:
            chunk["source"] = pdf_path.name

        # Save the intermediate results for notebook inspection.
        self.sections = sections
        self.chunks = chunks
        self.source_path = str(pdf_path)

        return {
            "pages": len(pages),
            "sections": len(sections),
            "chunks": len(chunks),
            "section_preview": sections[:5],
            "chunk_preview": chunks[:5],
        }

    def build_index(self, pdf_path=None, remove_toc=False):
        if pdf_path is not None:
            summary = self.process_document(
                pdf_path,
                remove_toc=remove_toc,
            )
        elif self.chunks:
            summary = {
                "pages": None,
                "sections": len(self.sections),
                "chunks": len(self.chunks),
            }
        else:
            raise ValueError(
                "Call process_document(pdf_path) first."
            )

        # Rebuild cleanly so old chunks from a previous document
        # are not accidentally mixed into the new index.
        self.vector_store.create_collection(reset=True)

        count = self.vector_store.add_chunks(self.chunks)

        summary["indexed_chunks"] = count
        return summary

    def retrieve(self, question, top_k=TOP_K, chapter=None):
        return self.retriever.retrieve(
            question=question,
            top_k=top_k,
            chapter=chapter,
        )

    def ask(self, question, top_k=TOP_K, chapter=None):
        results = self.retrieve(
            question=question,
            top_k=top_k,
            chapter=chapter,
        )

        answer = self.generator.generate(question, results)

        return {
            "question": question,
            "answer": answer,
            "sources": results,
        }
