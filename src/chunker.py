
import re

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks while preserving paragraph boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        # Split oversized paragraphs into pieces with overlap.
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)

                if end >= len(paragraph):
                    break
                start = end - overlap
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        prefix = current[-overlap:] if overlap and current else ""
        current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph

        # Keep the chunk within the size limit.
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size].strip())
            current = current[chunk_size - overlap:]

    if current:
        chunks.append(current.strip())

    return chunks


def split_lines_with_pages(lines, chunk_size=CHUNK_SIZE,
                           overlap=CHUNK_OVERLAP):
    """
    Merge consecutive source lines into paragraphs and chunks.
    Each line must be {"text": "...", "page": 37}.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    # Preserve the source page for every non-empty line.
    source_lines = []
    for item in lines:
        text = str(item.get("text", "")).strip()
        if text:
            source_lines.append({
                "text": text,
                "page": item.get("page"),
            })

    if not source_lines:
        return []

    # Join lines into paragraphs. Page changes alone do not end a paragraph.
    paragraphs = []
    current = []

    for item in source_lines:
        # Blank lines must be preserved in the input to detect paragraph
        # boundaries. If the extractor removes blank lines, adjacent lines
        # will instead be joined below.
        current.append(item)

    if current:
        paragraphs.append(current)

    # Convert source lines into text while retaining character-level
    # page provenance. This allows chunks to span pages accurately.
    text_parts = []
    char_pages = []

    for item in source_lines:
        part = item["text"] + "\n"
        text_parts.append(part)
        char_pages.extend([item["page"]] * len(part))

    full_text = "".join(text_parts).strip()
    char_pages = char_pages[:len(full_text)]

    # Prefer splitting near paragraph breaks, then line breaks, then spaces.
    chunks = []
    start = 0
    text_length = len(full_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Avoid very short chunks by seeking a good boundary.
            search_start = start + int(chunk_size * 0.6)
            boundary = full_text.rfind("\n\n", search_start, end)
            if boundary == -1:
                boundary = full_text.rfind("\n", search_start, end)
            if boundary == -1:
                boundary = full_text.rfind(" ", search_start, end)
            if boundary > start:
                end = boundary

        piece = full_text[start:end].strip()

        if piece:
            page_slice = char_pages[start:end]
            pages = [p for p in page_slice if p is not None]

            chunks.append({
                "text": piece,
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
            })

        if end >= text_length:
            break

        next_start = max(start + 1, end - overlap)

        # Avoid starting the next chunk in the middle of a word.
        while next_start < end and full_text[next_start].isalnum():
            next_start += 1

        start = next_start

    return chunks


def build_chunks(sections):
    """Build chunks with accurate page metadata when lines are available."""
    chunks = []

    for section_index, section in enumerate(sections):
        lines = section.get("lines")

        if lines:
            section_chunks = split_lines_with_pages(
                lines,
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        else:
            texts = split_text(
                section.get("text", ""),
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )

            section_chunks = [
                {
                    "text": text,
                    "page_start": section.get("page_start"),
                    "page_end": section.get("page_end"),
                }
                for text in texts
            ]

        for chunk_index, chunk in enumerate(section_chunks):
            chunks.append({
                "text": chunk["text"],
                "chapter": section.get("chapter", "General"),
                "section": section.get("section", "Unknown"),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section_index": section_index,
                "chunk_index": chunk_index,
            })

    return chunks
