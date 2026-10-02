"""
Configuration settings for the RAG application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
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

# Model Pricing (per 1,000,000 tokens)
EMBEDDING_PRICE_PER_M = 0.02
CHAT_INPUT_PRICE_PER_M = 0.15
CHAT_OUTPUT_PRICE_PER_M = 0.60

# Chunking Hyperparameters
TARGET_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
MIN_CHUNK_CHARS = 50

# Retrieval Hyperparameters
TOP_K = 7
DENSE_WEIGHT = 0.65
BM25_WEIGHT = 0.35
SIMILARITY_THRESHOLD = 0.25  # Lower bound for considering retrieved chunk relevant
