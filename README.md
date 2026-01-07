# SEC 10-K RAG Assistant

AI-powered assistant for drafting SEC Form 10-K Business (Item 1) and MD&A (Item 7) sections using Retrieval-Augmented Generation (RAG).

## Requirements

- **Python 3.9 - 3.13** (Python 3.14 may have compatibility issues with some dependencies)
- OpenAI API key

## Quick Start (For Reviewers)

The repository includes pre-downloaded 10-K filings and pre-built vector index. You only need to:

```bash
# 1. Clone the repository
git clone https://github.com/zeyuyuyu/sec-10k-rag-assistant.git
cd sec-10k-rag-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set OpenAI API key
export OPENAI_API_KEY=your-key-here

# 4. Start interactive chat
python main.py chat
```

## Testing the System

### Test Case 1: Generate Business and MD&A for NVIDIA

```bash
python main.py chat
```

Then enter:
```
Generate the Business and MD&A sections for NVDA's 2025 Form 10-K
```

**Expected behavior:**
1. System retrieves relevant context from NVDA's prior 10-K filing
2. Generates Item 1 (Business) section with source citations
3. Asks for financial data to complete MD&A

### Test Case 2: Provide Financial Data

When prompted for financial data, provide (in any format):

**Markdown table:**
```
| Metric | FY 2025 | FY 2024 |
|--------|---------|---------|
| Revenue | $130.5B | $60.9B |
| Revenue Growth | 114% | 126% |
| Operating Income | $81.0B | $32.9B |
| Net Income | $72.9B | $29.8B |
```

**Or plain text:**
```
Revenue: $130.5 billion (up 114% YoY)
Operating income: $81.0 billion
Net income: $72.9 billion
```

**Or HTML table:**
```html
<table>
<tr><th>Metric</th><th>FY 2025</th><th>FY 2024</th></tr>
<tr><td>Revenue</td><td>$130.5B</td><td>$60.9B</td></tr>
</table>
```

**Expected behavior:**
1. System parses the financial data
2. Generates Item 7 (MD&A) section incorporating the data
3. Shows Year-over-Year analysis table
4. Displays confidence indicator
5. Saves audit log

### Test Case 3: API Testing

```bash
# Start the server
python main.py serve

# In another terminal, test the API
curl http://localhost:8000/companies
curl -X POST "http://localhost:8000/chat/start?session_id=test1"
```

## Features

- ✅ RAG over SEC EDGAR 10-K filings (8 companies pre-loaded)
- ✅ Interactive financial data collection
- ✅ Generates legally-appropriate narrative text
- ✅ FastAPI backend + CLI interface
- ✅ Supports multiple input formats (Markdown, HTML, plain text)

## Supported Companies

| Ticker | Company |
|--------|---------|
| NVDA | NVIDIA Corporation |
| MSFT | Microsoft Corporation |
| KO | The Coca-Cola Company |
| NKE | NIKE, Inc. |
| AMZN | Amazon.com, Inc. |
| DASH | DoorDash, Inc. |
| TJX | The TJX Companies, Inc. |
| DRI | Darden Restaurants, Inc. |

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py chat` | Interactive chat mode (recommended) |
| `python main.py serve` | Start FastAPI server |
| `python main.py companies` | List available companies |
| `python main.py generate NVDA 2025` | Generate Business section only |
| `python main.py download --all` | Re-download 10-K filings (optional) |
| `python main.py index --rebuild` | Rebuild vector index (optional) |

## API Endpoints

Start server: `python main.py serve`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/companies` | GET | List companies |
| `/chat` | POST | Interactive chat |
| `/chat/start` | POST | Start new session |
| `/generate` | POST | Direct generation |
| `/sessions/{id}/audit` | GET | Get audit log |

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
When financial data with prior year comparisons is provided:
```
| Metric | Current Year | Prior Year | Change | % Change | Trend |
|--------|--------------|------------|--------|----------|-------|
| Revenue | 130.5B | 60.9B | +69.6B | +114.3% | 📈 |
```

### 4. Audit-Friendly Logging
All interactions logged to `data/audit_logs/` with:
- Session ID and timestamps
- Content hashes for integrity verification
- User inputs (raw and parsed)
- Generated outputs with source citations
- Confidence scores

## Project Structure

```
sec-10k-rag-assistant/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── README.md              # This file
├── src/
│   ├── config.py          # Configuration
│   ├── sec_downloader.py  # SEC EDGAR downloader
│   ├── document_processor.py # Document chunking & vectorization
│   ├── rag_engine.py      # RAG generation engine
│   ├── assistant.py       # Interactive assistant
│   ├── api.py             # FastAPI backend
│   ├── cli.py             # CLI interface
│   ├── citations.py       # Source citations & confidence
│   ├── yoy_analysis.py    # Year-over-year analysis
│   └── audit_logger.py    # Audit logging
└── data/
    ├── filings/           # Pre-downloaded 10-K files (8 companies)
    ├── vector_db/         # Pre-built FAISS index (1866 chunks)
    └── audit_logs/        # Audit trail logs
```

## Tech Stack

- Python 3.9+
- LangChain (RAG framework)
- OpenAI GPT-4 (LLM)
- FAISS (Vector database)
- FastAPI (Backend)
- Typer + Rich (CLI)

## License

MIT
