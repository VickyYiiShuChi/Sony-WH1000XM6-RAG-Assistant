
import re
from collections import Counter


PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$",
    re.IGNORECASE,
)


def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def remove_repeated_headers_footers(pages):
    """Remove short lines repeated near page edges."""
    counts = Counter()
    total_pages = len(pages)

    for page in pages:
        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        edge_lines = lines[:3] + lines[-3:]

        for line in edge_lines:
            if len(line) <= 100:
                counts[line.lower()] += 1

    threshold = max(4, int(total_pages * 0.25))

    repeated = {
        line
        for line, count in counts.items()
        if count >= threshold
    }

    cleaned = []

    for page in pages:
        kept_lines = []

        for line in page["text"].splitlines():
            line = line.strip()

            if not line:
                continue

            if PAGE_NUMBER_RE.fullmatch(line):
                continue

            if line.lower() in repeated:
                continue

            kept_lines.append(line)

        cleaned.append({
            "page": page["page"],
            "text": normalize_text("\n".join(kept_lines)),
        })

    return cleaned


def is_table_of_contents_page(text):
    lower = text.lower()

    if "table of contents" in lower:
        return True

    if re.search(r"^\s*contents\s*$", lower, re.MULTILINE):
        return True

    dotted_lines = re.findall(
        r"\.{3,}\s*\d+\s*$",
        text,
        re.MULTILINE,
    )

    return len(dotted_lines) >= 5


def clean_pages(pages, remove_toc=False):
    """
    Clean PDF pages.

    remove_toc defaults to False because TOC pages can contain
    useful navigation information. Enable it only after inspection.
    """
    pages = remove_repeated_headers_footers(pages)

    if remove_toc:
        pages = [
            page for page in pages
            if not is_table_of_contents_page(page["text"])
        ]

    return [
        page for page in pages
        if page["text"].strip()
    ]
