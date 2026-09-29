#!/usr/bin/env python3
"""
Corpus Extraction Script for RAG Project
Extracts clean, structured text from documents in data/sources/
and outputs data/corpus.json and data/sample.json.
"""

import json
import re
from pathlib import Path
import pymupdf  # PyMuPDF
from bs4 import BeautifulSoup

SOURCES_DIR = Path("data/sources")
CORPUS_PATH = Path("data/corpus.json")
SAMPLE_PATH = Path("data/sample.json")
MIN_CHARS = 50

# Regex patterns for running headers and page numbers
SECTION_HEADER_RE = re.compile(r"^\d+\.\d+\.?\s+[A-Z0-9\s,:()\-]+(?:\s+\d+)?$")
CHAPTER_HEADER_RE = re.compile(r"^CHAPTER\s+\d+\.\s+[A-Z0-9\s,:()\-]+$", re.IGNORECASE)


def clean_extracted_text(raw_text: str) -> str:
    """
    Cleans extracted text:
    - Removes running headers/footers and standalone page numbers
    - Handles line-end hyphenation (de-hyphenating split words)
    - Normalizes curly quotes/apostrophes and whitespace
    - Preserves code structure and lists
    """
    if not raw_text:
        return ""

    raw_lines = [ln.strip() for ln in raw_text.splitlines()]
    if not raw_lines:
        return ""

    # Remove running headers and page numbers from top of page
    # (Typically line 1 is page num or section header, line 2 is chapter title or section)
    while raw_lines and (
        raw_lines[0].isdigit()
        or SECTION_HEADER_RE.match(raw_lines[0])
        or CHAPTER_HEADER_RE.match(raw_lines[0])
    ):
        raw_lines.pop(0)

    # In case there's another header line
    if raw_lines and (
        raw_lines[0].isdigit()
        or SECTION_HEADER_RE.match(raw_lines[0])
        or CHAPTER_HEADER_RE.match(raw_lines[0])
    ):
        raw_lines.pop(0)

    # Remove standalone page numbers from the bottom of the page
    if raw_lines and raw_lines[-1].isdigit():
        raw_lines.pop(-1)

    if not raw_lines:
        return ""

    # De-hyphenate words broken across line breaks and clean whitespace
    processed_lines = []
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        if not line:
            if processed_lines and processed_lines[-1] != "":
                processed_lines.append("")
            i += 1
            continue

        # Check for hyphenation at line end connecting to next line
        if line.endswith("-") and i + 1 < len(raw_lines) and raw_lines[i + 1]:
            next_line = raw_lines[i + 1]
            # If next word starts with lowercase, merge the split word
            first_word_match = re.match(r"^([a-z]+)(\s+.*)?$", next_line)
            if first_word_match:
                prefix = line[:-1]
                suffix = first_word_match.group(1)
                remainder = first_word_match.group(2) or ""
                line = prefix + suffix + remainder
                i += 1

        processed_lines.append(line)
        i += 1

    # Normalize multiple newlines and spaces
    cleaned = "\n".join(processed_lines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_pdf(path: Path):
    """Extract text page-by-page from a PDF."""
    doc = pymupdf.open(path)
    for page_idx, page in enumerate(doc, start=1):
        raw_text = page.get_text()
        cleaned_text = clean_extracted_text(raw_text)
        yield page_idx, cleaned_text


def extract_text_file(path: Path):
    """Extract text from plain text or markdown files."""
    raw_text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in raw_text.splitlines()]
    cleaned_text = "\n".join(ln for ln in lines if ln)
    yield 1, cleaned_text


def extract_html(path: Path):
    """Extract clean text from an HTML file."""
    html_content = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_text = "\n".join(ln for ln in lines if ln)
    yield 1, cleaned_text


def main():
    if not SOURCES_DIR.exists():
        raise FileNotFoundError(f"Sources directory not found: {SOURCES_DIR}")

    entries = []
    near_empty_count = 0
    source_files = sorted(
        [p for p in SOURCES_DIR.iterdir() if p.is_file() and not p.name.startswith(".")]
    )

    print(f"Processing {len(source_files)} source files from {SOURCES_DIR}...")

    for path in source_files:
        ext = path.suffix.lower()
        if ext == ".pdf":
            page_generator = extract_pdf(path)
        elif ext in {".txt", ".md"}:
            page_generator = extract_text_file(path)
        elif ext in {".html", ".htm"}:
            page_generator = extract_html(path)
        else:
            print(f"Skipping unsupported file type: {path.name}")
            continue

        file_entries = 0
        for page_num, text in page_generator:
            char_count = len(text)
            if char_count < MIN_CHARS:
                near_empty_count += 1

            entries.append({
                "source": path.name,
                "page": page_num,
                "char_count": char_count,
                "text": text,
            })
            file_entries += 1

        print(f"  Extracted {file_entries} pages from {path.name}")

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    SAMPLE_PATH.write_text(
        json.dumps(entries[:10], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_entries = len(entries)
    total_chars = sum(e["char_count"] for e in entries)
    unique_files = len({e["source"] for e in entries})
    avg_chars = total_chars // max(total_entries, 1)

    print("\n--- Corpus Statistics ---")
    print(f"source files: {unique_files}")
    print(f"total entries: {total_entries}")
    print(f"total characters: {total_chars}")
    print(f"avg characters/entry: {avg_chars}")
    print(f"near-empty entries (<{MIN_CHARS} chars): {near_empty_count}")


if __name__ == "__main__":
    main()
