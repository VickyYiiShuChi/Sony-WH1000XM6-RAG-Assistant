
import re


# Sony WH-1000XM6 manual
# PDF pages 3-6 contain the table of contents.
TOC_PAGES = {3, 4, 5, 6}

# On PDF page 2, the Quick Start Guide ends before this heading.
TOC_START_MARKER = "Getting started"


# Known headings from the manual.
KNOWN_HEADINGS = {
    "quick start guide",
    "find your fit",
    "wearing the headset",
    "download the app, and set up the headset",
    "turning on the headset",
    "turning off the headset",
    "establishing a bluetooth connection",
    "establishing a bluetooth® connection",
    "establishing a bluetooth connection manually",
    "operating the headset",
    "related information",
    "getting started",
    "what you can do with the bluetooth function",
    "about the voice guidance",
    "supplied accessories",
    "included items",
    "setting the headset in the carrying case",
    "parts and controls",
    "location and function of parts",
    "about the indicator",
    "about the touch sensor control panel",
    "using quick access",
    "power/charging",
    "charging the headset",
    "available operating time",
    "checking the remaining battery charge",
    "making connections",
    "how to make a wireless connection to bluetooth devices",
    "easy setup with app",
    "connecting with the “sony | sound connect” app",
    "connecting with the sony | sound connect app",
    "android smartphone",
    "pairing and connecting with an android smartphone",
    "pairing and connecting with an android smartphone using le audio",
    "connecting to a paired android smartphone",
    "iphone (ios devices)",
    "pairing and connecting with an iphone",
    "connecting to a paired iphone",
    "computers",
    "pairing and connecting with a computer (windows® 11)",
    "pairing and connecting with a computer (windows 11)",
    "pairing and connecting with a computer (windows 10)",
    "pairing and connecting with a computer (mac)",
    "connecting to a paired computer (windows 11)",
    "connecting to a paired computer (windows 10)",
    "connecting to a paired computer (mac)",
    "other bluetooth devices",
    "pairing and connecting with a bluetooth device",
    "connecting to a paired bluetooth device",
    "multipoint connection",
    "connecting the headset to 2 devices simultaneously (multipoint connection)",
    "switching audio between multiple devices with “audio switch”",
    "sharing music with “auracast™ broadcast audio”",
    "disconnecting bluetooth connection (after use)",
    "using the supplied headphone cable",
    "listening to music",
    "listening to music via a bluetooth connection",
    "listening to music from a device via a bluetooth connection",
    "controlling the audio device (bluetooth connection)",
    "about 360 reality audio",
    "enjoying seamless playback with speakers (auto switch)",
    "noise canceling function",
    "what is noise canceling?",
    "using the noise canceling function",
    "always automatically optimizing the noise canceling performance to suit the wearer and usage environment",
    "listening to ambient sound",
    "listening to ambient sound during music playback (ambient sound mode)",
    "automatically adjusting the level of ambient sound capture according to the ambient noise level (auto ambient sound)",
    "listening to ambient sound quickly (quick attention mode)",
    "speaking with someone while wearing the headset (speak-to-chat)",
    "sound quality mode",
    "about the sound quality mode",
    "supported codecs",
    "about the dsee extreme function",
    "using 360 upmix for cinema",
    "making phone calls",
    "talking on the phone",
    "receiving a call",
    "making a call",
    "functions for a phone call",
    "making a video call on your computer",
    "using the voice assist function",
    "using voice control",
    "using google assistant",
    "using tencent xiaowei",
    "using the voice assist function (google app)",
    "using the voice assist function (siri)",
    "using the apps",
    "what you can do with the “sony | sound connect” app",
    "installing the “sony | sound connect” app",
    "accessing support information from the “sony | sound connect” app",
    "how to keep the software up-to-date (for comfortable use of the headset)",
    "what you can do with partner services",
    "important information",
    "precautions",
    "to prevent burning or malfunction due to getting wet",
    "licenses",
    "trademarks",
    "customer support websites",
    "troubleshooting",
    "resetting the headset",
    "initializing the headset to restore factory settings",
    "specifications",
}


# Links under "Related information" are cross-references,
# not the main content of the current section.
RELATED_INFORMATION_LINKS = {
    "wearing the headset",
    "establishing a bluetooth connection manually",
    "charging the headset",
    "connecting to a paired android smartphone",
    "connecting to a paired iphone",
}


def normalize_heading(line):
    """Normalize text for case-insensitive heading comparison."""
    line = re.sub(r"\s+", " ", line.strip())
    return line.rstrip(":").strip().casefold()


def is_page_number(line):
    """Detect standalone printed page numbers."""
    return bool(re.fullmatch(r"\d{1,3}", line.strip()))


def is_repeated_header_or_footer(line):
    """Remove recurring headers, footers, and document codes."""
    text = line.strip()

    if not text or is_page_number(text):
        return True

    lower = text.casefold()

    if lower in {
        "help guide",
        "wireless noise canceling stereo headset",
        "wh-1000xm6",
    }:
        return True

    if re.fullmatch(
        r"5-063-141-11\(\d+\)\s*copyright\s*2025\s*sony corporation",
        text,
        re.IGNORECASE,
    ):
        return True

    return False


def is_heading(line):
    """Return True if the normalized line matches a known heading."""
    normalized = normalize_heading(line)

    if not normalized or len(normalized) > 180:
        return False

    return normalized in KNOWN_HEADINGS


def clean_page_text(page):
    """
    Clean one PDF page while preserving useful content.

    Page 1: preserve Quick Start Guide content.
    Page 2: keep content before the table of contents.
    Pages 3-6: skip the table of contents.
    Page 7 onward: preserve manual body text.
    """
    page_number = page["page"]
    text = page["text"]

    if page_number in TOC_PAGES:
        return ""

    lines = text.splitlines()
    kept_lines = []

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            continue

        # Page 2 contains both Quick Start Guide content and TOC.
        if page_number == 2:
            if normalize_heading(line) == normalize_heading(
                TOC_START_MARKER
            ):
                break

        if is_repeated_header_or_footer(line):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def extract_document_structure(pages):
    """
    Convert parsed PDF pages into sections.

    Each section contains:
    - chapter
    - section
    - text
    - page_start
    - page_end
    """
    sections = []

    chapter = "General"
    section_title = "Introduction"

    current_lines = []
    current_pages = []
    in_related_information = False

    def save_section():
        nonlocal current_lines, current_pages

        content = "\n".join(current_lines).strip()

        if content and current_pages:
            sections.append({
                "chapter": chapter,
                "section": section_title,
                "text": content,
                "page_start": min(current_pages),
                "page_end": max(current_pages),
                "lines": [
                    {"text": line, "page": page_number}
                    for line, page_number in zip(current_lines, current_pages)
                ],
            })

        current_lines = []
        current_pages = []

    for page in pages:
        page_number = page["page"]
        cleaned_text = clean_page_text(page)

        if not cleaned_text:
            continue

        for raw_line in cleaned_text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()

            if not line:
                continue

            normalized = normalize_heading(line)

            # Start or end the cross-reference block.
            if normalized == "related information":
                save_section()
                section_title = line
                in_related_information = True
                continue

            if in_related_information:
                # Skip known cross-reference entries.
                if normalized in RELATED_INFORMATION_LINKS:
                    continue

                # A new recognized heading means the reference block ended.
                if is_heading(line):
                    in_related_information = False
                else:
                    # Avoid accidentally adding a link to the previous section.
                    continue

            # Keep recognized headings separate from body text.
            if is_heading(line):
                save_section()
                section_title = line
                continue

            current_lines.append(line)
            current_pages.append(page_number)

    save_section()

    return sections
