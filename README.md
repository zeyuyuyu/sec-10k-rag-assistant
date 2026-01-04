# SEC 10-K RAG Assistant

AI-powered assistant for drafting SEC Form 10-K Business (Item 1) and MD&A (Item 7) sections.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set OpenAI API key
export OPENAI_API_KEY=your-key-here

# 3. Download filings and build index
python main.py download --all
python main.py index

# 4. Start interactive chat
python main.py chat
```

## Features

- RAG over SEC EDGAR 10-K filings
- Interactive financial data collection
- Generates legally-appropriate narrative
- FastAPI backend + CLI interface

## Supported Companies

NVDA, MSFT, KO, NKE, AMZN, DASH, TJX, DRI

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py download --all` | Download all 10-K filings |
| `python main.py download NVDA` | Download specific company |
| `python main.py index` | Build vector index |
| `python main.py chat` | Interactive chat mode |
| `python main.py generate NVDA 2024` | Generate Business section |
| `python main.py serve` | Start FastAPI server |
| `python main.py companies` | List available companies |

## API Endpoints

Start server: `python main.py serve`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/companies` | GET | List companies |
| `/chat` | POST | Interactive chat |
| `/chat/start` | POST | Start new session |
| `/generate` | POST | Direct generation |
| `/reset` | POST | Reset session |

## Example: Providing Financial Data

The assistant will ask for financial data to complete MD&A. You can provide data in any format:

**As a table:**
```
| Metric | FY 2024 | FY 2023 |
|--------|---------|---------|
| Revenue | $37.2B | $33.9B |
| Revenue Growth | 9.7% | 16.9% |
| Operating Income | $1.9B | $1.1B |
```

**As plain text:**
```
Revenue: $37.2 billion (up 9.7% YoY)
Operating income: $1.9 billion
Free cash flow: $3.4 billion
```

## Project Structure

```
lvsuo/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── src/
│   ├── config.py        # Configuration
│   ├── sec_downloader.py # SEC EDGAR downloader
│   ├── document_processor.py # Document processing
│   ├── rag_engine.py    # RAG generation engine
│   ├── assistant.py     # Interactive assistant
│   ├── api.py          # FastAPI backend
│   └── cli.py          # CLI interface
└── data/
    ├── filings/        # Downloaded 10-K files
    └── vector_db/      # FAISS vector store
```

## Enhanced Features (Optional Enhancements)

All optional enhancements from the assignment are implemented:

### 1. Source Citations
Each generated paragraph includes source references `[Source N]` linked to prior 10-K filings.

### 2. Confidence Indicators
Every generation includes a confidence assessment:
```
Confidence Assessment 🟢
| Metric | Score |
|--------|-------|
| Overall Confidence | HIGH (85%) |
| Data Coverage | 90% |
| Source Quality | 80% |
```

### 3. Year-over-Year Analysis
When financial data with prior year comparisons is provided, the system generates a YoY analysis table:
```
| Metric | Current Year | Prior Year | Change | % Change | Trend |
|--------|--------------|------------|--------|----------|-------|
| Revenue | 130.5B | 60.9B | +69.6B | +114.3% | 📈 |
```

### 4. Audit-Friendly Logging
All interactions are logged to `data/audit_logs/` with:
- Session ID and timestamps
- Content hashes for integrity verification
- User inputs (raw and parsed)
- Generated outputs
- Source citations used
- Confidence scores

## Tech Stack

- Python 3.9+
- LangChain (RAG framework)
- OpenAI GPT-4 (LLM)
- FAISS (Vector database)
- FastAPI (Backend)
- Typer + Rich (CLI)

## Project Structure

```
lvsuo/
├── main.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py           # Configuration
│   ├── sec_downloader.py   # SEC EDGAR downloader
│   ├── document_processor.py # Document processing
│   ├── rag_engine.py       # RAG generation engine
│   ├── assistant.py        # Interactive assistant
│   ├── api.py              # FastAPI backend
│   ├── cli.py              # CLI interface
│   ├── citations.py        # Source citations & confidence
│   ├── yoy_analysis.py     # Year-over-year analysis
│   └── audit_logger.py     # Audit logging
└── data/
    ├── filings/            # Downloaded 10-K files
    ├── vector_db/          # FAISS vector store
    └── audit_logs/         # Audit trail logs
```

## License

MIT
