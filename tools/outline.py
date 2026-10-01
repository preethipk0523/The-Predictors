"""
Document Outline & Table of Contents Tool (Extension)
Allows the agent to quickly inspect chapter names and starting pages.
"""

from typing import Dict, Any, List

CHAPTERS = [
    {"chapter": 1, "title": "Introduction", "start_page": 1, "pdf_page": 7},
    {"chapter": 2, "title": "A trip to Jupyter", "start_page": 9, "pdf_page": 15},
    {"chapter": 3, "title": "Three kinds of atomic data", "start_page": 13, "pdf_page": 19},
    {"chapter": 4, "title": "Memory pictures", "start_page": 25, "pdf_page": 31},
    {"chapter": 5, "title": "Calculations", "start_page": 31, "pdf_page": 37},
    {"chapter": 6, "title": "Scales of measure", "start_page": 43, "pdf_page": 49},
    {"chapter": 7, "title": "Three kinds of aggregate data", "start_page": 53, "pdf_page": 59},
    {"chapter": 8, "title": "Arrays in Python (1 of 2)", "start_page": 61, "pdf_page": 67},
    {"chapter": 9, "title": "Arrays in Python (2 of 2)", "start_page": 73, "pdf_page": 79},
    {"chapter": 10, "title": "Interpreting Data", "start_page": 89, "pdf_page": 95},
    {"chapter": 11, "title": "Assoc. arrays in Python (1 of 3)", "start_page": 103, "pdf_page": 109},
    {"chapter": 12, "title": "Assoc. arrays in Python (2 of 3)", "start_page": 109, "pdf_page": 115},
    {"chapter": 13, "title": "Assoc. arrays in Python (3 of 3)", "start_page": 123, "pdf_page": 129},
    {"chapter": 14, "title": "Loops", "start_page": 135, "pdf_page": 141},
    {"chapter": 15, "title": "EDA: univariate", "start_page": 145, "pdf_page": 151},
    {"chapter": 16, "title": "Tables in Python (1 of 3)", "start_page": 169, "pdf_page": 175},
    {"chapter": 17, "title": "Tables in Python (2 of 3)", "start_page": 177, "pdf_page": 183},
    {"chapter": 18, "title": "Tables in Python (3 of 3)", "start_page": 187, "pdf_page": 193},
    {"chapter": 19, "title": "EDA: bivariate (1 of 2)", "start_page": 193, "pdf_page": 199},
    {"chapter": 20, "title": "EDA: bivariate (2 of 2)", "start_page": 201, "pdf_page": 207},
    {"chapter": 21, "title": "Branching", "start_page": 211, "pdf_page": 217},
    {"chapter": 22, "title": "Functions (1 of 2)", "start_page": 223, "pdf_page": 229},
    {"chapter": 23, "title": "Functions (2 of 2)", "start_page": 235, "pdf_page": 241},
    {"chapter": 24, "title": "Recoding and transforming", "start_page": 243, "pdf_page": 249},
    {"chapter": 25, "title": "Machine Learning: concepts", "start_page": 255, "pdf_page": 261},
    {"chapter": 26, "title": "Classification: concepts", "start_page": 261, "pdf_page": 267},
    {"chapter": 27, "title": "Decision trees (1 of 2)", "start_page": 269, "pdf_page": 275},
    {"chapter": 28, "title": "Decision trees (2 of 2)", "start_page": 275, "pdf_page": 281},
    {"chapter": 29, "title": "Evaluating a classifier", "start_page": 283, "pdf_page": 289},
]


def list_chapters() -> Dict[str, Any]:
    """Return the table of contents and chapter page mapping for The Crystal Ball manual."""
    summary_lines = [
        f"Chapter {c['chapter']}: {c['title']} (Starts at book page {c['start_page']}, PDF page {c['pdf_page']})"
        for c in CHAPTERS
    ]
    return {
        "total_chapters": len(CHAPTERS),
        "chapters": CHAPTERS,
        "formatted": "\n".join(summary_lines)
    }
