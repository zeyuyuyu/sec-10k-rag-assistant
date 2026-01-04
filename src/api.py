"""FastAPI backend for 10-K RAG Assistant."""
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from src.assistant import TenKAssistant, create_assistant
from src.config import TARGET_COMPANIES


app = FastAPI(
    title="SEC 10-K RAG Assistant API",
    description="AI-powered assistant for drafting SEC Form 10-K Business and MD&A sections",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage (in production, use Redis or database)
sessions: Dict[str, TenKAssistant] = {}


class ChatRequest(BaseModel):
    """Chat request model."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    state: str
    ticker: Optional[str] = None
    fiscal_year: Optional[str] = None


class GenerateRequest(BaseModel):
    """Direct generation request model."""
    ticker: str
    fiscal_year: str
    financial_data: Optional[Dict] = None
    business_inputs: Optional[Dict] = None


class GenerateResponse(BaseModel):
    """Generation response model."""
    business_section: Optional[str] = None
    mda_section: Optional[str] = None
    missing_data_questions: Optional[str] = None
    # Enhanced features
    citations: Optional[list] = None
    confidence: Optional[Dict] = None
    yoy_analysis: Optional[list] = None
    audit_log_path: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SEC 10-K RAG Assistant API",
        "docs": "/docs",
        "available_endpoints": [
            "/companies - List available companies",
            "/chat - Interactive chat endpoint",
            "/generate - Direct generation endpoint",
            "/reset - Reset conversation session",
        ]
    }


@app.get("/companies")
async def list_companies():
    """List available companies for 10-K generation."""
    return {
        "companies": [
            {
                "ticker": ticker,
                "name": info["name"],
                "cik": info["cik"],
            }
            for ticker, info in TARGET_COMPANIES.items()
        ]
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Interactive chat endpoint."""
    session_id = request.session_id
    
    # Get or create session
    if session_id not in sessions:
        sessions[session_id] = create_assistant()
    
    assistant = sessions[session_id]
    
    try:
        response = assistant.process_message(request.message)
        return ChatResponse(
            response=response,
            state=assistant.context.state.value,
            ticker=assistant.context.ticker,
            fiscal_year=assistant.context.fiscal_year,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/start", response_model=ChatResponse)
async def start_chat(session_id: str):
    """Start a new chat session."""
    sessions[session_id] = create_assistant()
    assistant = sessions[session_id]
    
    response = assistant._get_initial_response()
    assistant._add_message("assistant", response)
    
    return ChatResponse(
        response=response,
        state=assistant.context.state.value,
    )


@app.post("/reset")
async def reset_session(session_id: str):
    """Reset a conversation session."""
    if session_id in sessions:
        sessions[session_id].reset()
        return {"message": "Session reset successfully"}
    return {"message": "Session not found, creating new session"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_direct(request: GenerateRequest):
    """Direct generation endpoint (non-interactive) with enhanced features."""
    ticker = request.ticker.upper()
    
    if ticker not in TARGET_COMPANIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ticker: {ticker}. Available: {list(TARGET_COMPANIES.keys())}"
        )
    
    assistant = create_assistant()
    assistant.context.ticker = ticker
    assistant.context.company_name = TARGET_COMPANIES[ticker]["name"]
    assistant.context.fiscal_year = request.fiscal_year
    
    response = GenerateResponse()
    
    try:
        # Generate Business section with citations and confidence
        business_text, business_meta = assistant.rag_engine.generate_business_section(
            ticker,
            request.fiscal_year,
            include_citations=True,
        )
        response.business_section = business_text
        response.citations = business_meta.get("citations", [])
        response.confidence = business_meta.get("confidence", {})
        
        # Generate MD&A if financial data provided
        if request.financial_data:
            mda_text, mda_meta = assistant.rag_engine.generate_mda_section(
                ticker,
                request.fiscal_year,
                request.financial_data,
                include_citations=True,
                include_yoy_analysis=True,
            )
            response.mda_section = mda_text
            response.citations = mda_meta.get("citations", [])
            response.confidence = mda_meta.get("confidence", {})
            response.yoy_analysis = mda_meta.get("yoy_analysis", [])
        else:
            # Return questions for missing data
            response.missing_data_questions = assistant.rag_engine.ask_clarifying_questions(
                ticker,
                request.fiscal_year,
            )
        
        # Save and return audit log path
        response.audit_log_path = assistant.rag_engine.save_audit_log()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return response


@app.get("/sessions/{session_id}/audit")
async def get_audit_log(session_id: str):
    """Get audit log summary for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id].rag_engine.get_audit_summary()


@app.get("/sessions/{session_id}/content")
async def get_generated_content(session_id: str):
    """Get all generated content for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "sections": sessions[session_id].get_generated_content(),
        "context": {
            "ticker": sessions[session_id].context.ticker,
            "company_name": sessions[session_id].context.company_name,
            "fiscal_year": sessions[session_id].context.fiscal_year,
            "state": sessions[session_id].context.state.value,
        }
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

