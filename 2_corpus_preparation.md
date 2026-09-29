# Assignment 0: Corpus Preparation

> **Goal:** Extract clean text from your corpus documents so it's ready to use on Day 1 of the hackathon.
>
> **Deadline:** before Kaggle week

---

## Why this matters

On Day 1, you'll learn how to build a RAG pipeline. The first step of that pipeline is having clean text from your documents. Text extraction can be tricky, PDFs have weird text ordering, headers repeat on every page, tables get mangled, web pages have navigation noise. Doing this now means you arrive on Day 1 with data ready to go, instead of spending build time fighting formatting issues.

---

## Setup

### GitHub Repository

> ⚠️ Do not create the repo before your group is validated and your private channel exists.

Once your team's private channel has been created, create a **private** GitHub repository for your project:

- Add **bachtn** as a collaborator
- This repo will be used for the entire hackathon, all your code goes here
- Add a `.gitignore` that excludes `.venv/`, `.env`, and `data/`
- Add a `README.md` with your team name and members (you'll fill in the project description, getting started guide, etc. during the hackathon)
- Organize your repo like this:

```
your-team-name/
├── README.md
├── .gitignore
├── scripts/
│   ├── extract.py
│   └── EXTRACTION_REPORT.md
├── data/                  ← gitignored, uploaded to Teams only
│   ├── sources/           ← raw documents (PDFs, HTML, txt, etc.)
│   ├── corpus.json
│   └── sample.json
└── app/                   ← created during the hackathon
```

> ⚠️ The `data/` folder must be in `.gitignore`. Documents and JSON files go to your Teams channel, not GitHub. The tree above shows your local layout; `data/` is gitignored and shared via Teams, not committed to GitHub.

### Environment

Use **uv** for managing your Python environment. Requires **Python 3.12**.

```bash
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install pymupdf pdfplumber beautifulsoup4   # see "Libraries by format" below
```

---

## The task

> 💡 Don't hesitate to use GenAI tools (ChatGPT, Copilot) to help write the script.

First, place your corpus documents in `data/sources/`. Then write a Python script (`scripts/extract.py`) that:

1. Reads all documents from `data/sources/` (page-by-page for PDFs, one entry per file for other formats)
2. Saves `data/corpus.json` with metadata
3. Saves `data/sample.json` with the first 10 entries (for quick review)
4. Prints corpus stats:
   - Number of source files
   - Total entries
   - Total characters
   - Average characters per entry
   - Number of empty or near-empty entries (< 50 characters)

### Clarifications

- **`char_count`** is `len(text)` measured on the cleaned text (after the cleanup below), not the raw text.
- **Encoding:** read and write all files as UTF-8. This matters for accented French text and symbols.
- **Empty / near-empty entries (< 50 characters):** keep them in `corpus.json`, but count them and flag the number in `EXTRACTION_REPORT.md`. Don't silently drop data.
- **Scanned or image-only PDFs:** if a PDF has no selectable text, extraction returns empty. Note it in your report. OCR is out of scope for this assignment.
- **Multi-column PDFs:** text can come out interleaved. If a source is badly mangled, note it and consider a cleaner source.

### Expected output format

Your `corpus.json` (and `sample.json`, same format). For PDFs, one entry per page. For other formats, one entry per file (set `page` to 1):

```json
[
  {
    "source": "annual_report_2024.pdf",
    "page": 1,
    "char_count": 2847,
    "text": "The extracted text from page 1..."
  },
  {
    "source": "guidelines.pdf",
    "page": 2,
    "char_count": 3102,
    "text": "The extracted text from page 2..."
  },
  {
    "source": "faq.html",
    "page": 1,
    "char_count": 1956,
    "text": "The extracted text from the web page..."
  }
]
```

### Libraries by format

**PDFs (most common):**
- **PyMuPDF** (`uv pip install pymupdf`), fast, good general-purpose text extraction. Use this by default.
- **pdfplumber** (`uv pip install pdfplumber`), better for documents with tables. If your corpus has important tables that come out garbled with PyMuPDF, try pdfplumber instead.

**Web pages:** Use **BeautifulSoup** (`uv pip install beautifulsoup4`) to parse HTML and extract text content.

**Plain text / Markdown:** Read directly with Python, no library needed.

### Text cleanup

Raw extracted text is messy. At minimum, clean up:
- Extra whitespace (multiple spaces, blank lines)
- **PDFs:** repeated page headers/footers, line breaks within paragraphs (PDFs often break lines mid-sentence)
- **HTML:** navigation bars, sidebars, scripts, and other non-content elements

Don't aim for perfection, but don't skip this step either. The quality of your extracted text directly affects how well your RAG system will work later. Garbage in = garbage out.

### Starter script

A minimal skeleton to build on. It handles PDFs and plain text; add HTML parsing with BeautifulSoup if your corpus needs it.

```python
# scripts/extract.py
import json
from pathlib import Path
import fitz  # PyMuPDF

SOURCES = Path("data/sources")
MIN_CHARS = 50

def clean(text: str) -> str:
    # collapse whitespace; add your own header/footer removal
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)

def extract_pdf(path):
    doc = fitz.open(path)
    for i, page in enumerate(doc, start=1):
        yield i, clean(page.get_text())

def extract_text_file(path):
    yield 1, clean(path.read_text(encoding="utf-8"))

def main():
    entries, near_empty = [], 0
    for path in sorted(SOURCES.iterdir()):
        ext = path.suffix.lower()
        if ext == ".pdf":
            pages = extract_pdf(path)
        elif ext in {".txt", ".md"}:
            pages = extract_text_file(path)
        else:
            continue  # add .html handling with BeautifulSoup if needed
        for page, text in pages:
            if len(text) < MIN_CHARS:
                near_empty += 1        # keep it, but count it
            entries.append({
                "source": path.name,
                "page": page,
                "char_count": len(text),
                "text": text,
            })

    Path("data/corpus.json").write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("data/sample.json").write_text(
        json.dumps(entries[:10], ensure_ascii=False, indent=2), encoding="utf-8")

    chars = sum(e["char_count"] for e in entries)
    files = len({e["source"] for e in entries})
    print(f"source files: {files}")
    print(f"total entries: {len(entries)}")
    print(f"total characters: {chars}")
    print(f"avg characters/entry: {chars // max(len(entries), 1)}")
    print(f"near-empty entries (<{MIN_CHARS} chars): {near_empty}")

if __name__ == "__main__":
    main()
```

---

## Checklist

Before submitting, verify:

- [ ] Your script runs on your full corpus without errors
- [ ] `corpus.json` contains all entries from your corpus
- [ ] `sample.json` contains the first 10 entries
- [ ] Each entry has `source`, `page`, `char_count`, and `text` fields
- [ ] The text is readable (spot-check a few entries)
- [ ] Empty or near-empty entries (< 50 characters) are flagged in your `EXTRACTION_REPORT.md`

---

## Submission

**On your GitHub repo:**
- `scripts/extract.py`, your extraction script
- `scripts/EXTRACTION_REPORT.md`, issues encountered during extraction (e.g. garbled tables, scanned pages, encoding problems, repeated headers) and how you handled them. If extraction went smoothly, say so.

**On your team's private Teams channel:**

1. Upload `corpus.json` and `sample.json` to a `data/` folder in your channel's Files tab
2. Post a message with:
   - Link to your GitHub repo
   - Your corpus stats (copy-paste the script output)
