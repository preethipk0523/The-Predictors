"""
Configuration settings for the RAG application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env from project root
load_dotenv(BASE_DIR / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"

CORPUS_PATH = DATA_DIR / "corpus.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
BM25_PATH = DATA_DIR / "bm25.pkl"
QUESTIONS_PATH = DATA_DIR / "questions.json"
COST_LOG_PATH = DATA_DIR / "cost_tracker.json"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_DIM = 1536