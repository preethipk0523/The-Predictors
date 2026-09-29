# Assignment 0: Corpus Extraction Report

## 1. Corpus Overview
- **Source Document**: `crystal1 (1).pdf`
- **Title**: *The Crystal Ball Instruction Manual (Volume One: Introduction to Data Science, Version 1.2)*
- **Author**: Stephen Davies, Ph.D., Department of Computer Science, University of Mary Washington
- **Topic**: Foundational data science textbook covering data types (atomic, aggregate), scales of measurement, NumPy arrays, Pandas Series/DataFrames, exploratory data analysis, plotting, and machine learning fundamentals (decision trees, evaluation metrics).
- **Target Location**: `data/sources/crystal1 (1).pdf`

---

## 2. Corpus Statistics
The extraction was executed via `scripts/extract.py` and yielded the following verified statistics:

| Metric | Value |
| :--- | :--- |
| **Source files** | 1 |
| **Total entries (pages)** | 314 |
| **Total characters (cleaned)** | 382,242 |
| **Average characters per entry** | 1,217 |
| **Near-empty entries (< 50 characters)** | 16 |

### Analysis of Near-Empty Entries
All 16 near-empty entries correspond to intentionally blank pages or structural divider pages between major chapters in the published PDF:
- **Pages**: 2, 14, 66, 94, 140, 150, 182, 206, 228, 260, 294, 300, 311, 312, 313, 314
- **Handling**: In strict adherence to assignment requirements, these pages are retained in `data/corpus.json` with their exact page metadata (`char_count: 0`, `text: ""`) so that 1-to-1 page number parity is maintained with the original physical document.

---

## 3. Extraction Methodology & Text Cleaning

### A. Parser Selection
- **PyMuPDF (`pymupdf`)**: Chosen for high extraction speed, accurate coordinate detection, and preservation of text flow in academic textbooks.

### B. Issues Encountered & Solutions
1. **Running Headers & Section Breadcrumbs**:
   - *Issue*: Pages contained running headers at the top (e.g., `CHAPTER 1. INTRODUCTION`, `1.1. DEFINING DATA SCIENCE`, `5.2. STRING OPERATIONS 37`). If left in the text, these repeated headers would inject false keywords and distort semantic embeddings.
   - *Solution*: Implemented regex pattern matching (`SECTION_HEADER_RE`, `CHAPTER_HEADER_RE`) targeting the top 2 lines of each page to cleanly strip running headers while keeping genuine chapter opening titles intact.

2. **Standalone Page Numbers**:
   - *Issue*: Page numbers appeared both at the top (adjacent to headers) and bottom of pages (e.g., single digits/integers).
   - *Solution*: Identified and stripped leading/trailing standalone digit lines.

3. **Hyphenated Line Breaks**:
   - *Issue*: Justified paragraph text frequently broke words across lines with a trailing hyphen (e.g., `indepen-\ndent`, `be-\ncause`).
   - *Solution*: Implemented de-hyphenation logic: when a line ends with a hyphen and the following line begins with a lowercase letter, the split token is stitched back together into a single unified word.

4. **Unicode Quotes & Symbols**:
   - *Issue*: The PDF uses typographic quotation marks (`\u2019`, `\u201c`, `\u201d`) and special dashes.
   - *Solution*: Encoded files with UTF-8 (`ensure_ascii=False`) to preserve full unicode fidelity without corruption.

5. **Code Blocks & Tables**:
   - *Issue*: Aggressive line collapsing would destroy indentation and line separation in Python code blocks and tabular listings.
   - *Solution*: Preserved code line structure while collapsing extraneous multiple blank lines (`\n{3,}` into `\n\n`) and normalizing intra-line whitespace.

---

## 4. Deliverables Verification Checklist
- [x] Extraction script runs cleanly on the corpus (`scripts/extract.py`)
- [x] `data/corpus.json` generated containing all 314 pages with `source`, `page`, `char_count`, and `text`
- [x] `data/sample.json` generated with the first 10 entries for quick inspection
- [x] All 16 empty/near-empty entries cataloged and explained
- [x] UTF-8 encoding maintained throughout
- [x] Spot-checks conducted on multiple chapters confirming high text readability and accurate code snippets
