
import fitz


def extract_pdf_pages(pdf_path):
    """
    Extract selectable text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries containing:
        - page: 1-based PDF page number
        - text: extracted page text
    """
    pages = []

    with fitz.open(pdf_path) as pdf:
        if len(pdf) == 0:
            raise ValueError("The PDF contains no pages.")

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)

            # Normalize line endings without removing paragraph structure.
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    if not any(page["text"] for page in pages):
        raise ValueError(
            "No selectable text found. The PDF may require OCR."
        )

    return pages


def print_page_preview(pages, start_page=1, end_page=5):
    """Print extracted text to inspect reading order and page boundaries."""
    for page in pages:
        if start_page <= page["page"] <= end_page:
            print(f"\n{'=' * 20} PAGE {page['page']} {'=' * 20}")
            print(page["text"][:3000])


if __name__ == "__main__":
    from src.config import DATA_DIR

    pdf_path = DATA_DIR / "Sony_WH1000XM6_UserManual.pdf"
    extracted_pages = extract_pdf_pages(pdf_path)

    print(f"Total PDF pages: {len(extracted_pages)}")
    print_page_preview(extracted_pages, 1, 5)
