"""Configuration settings for the SEC 10-K RAG Assistant."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
FILINGS_DIR = DATA_DIR / "filings"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
FILINGS_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model settings
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# RAG settings
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 8

# Target companies for 10-K filings
TARGET_COMPANIES = {
    "NVDA": {"name": "NVIDIA Corporation", "cik": "0001045810"},
    "MSFT": {"name": "Microsoft Corporation", "cik": "0000789019"},
    "KO": {"name": "The Coca-Cola Company", "cik": "0000021344"},
    "NKE": {"name": "NIKE, Inc.", "cik": "0000320187"},
    "AMZN": {"name": "Amazon.com, Inc.", "cik": "0001018724"},
    "DASH": {"name": "DoorDash, Inc.", "cik": "0001792789"},
    "TJX": {"name": "The TJX Companies, Inc.", "cik": "0000109198"},
    "DRI": {"name": "Darden Restaurants, Inc.", "cik": "0000940944"},
}

# SEC EDGAR API settings
SEC_BASE_URL = "https://www.sec.gov"
SEC_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) SEC-RAG-Assistant contact@example.com"

