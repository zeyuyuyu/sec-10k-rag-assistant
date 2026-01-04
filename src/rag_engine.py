"""RAG Engine for 10-K generation."""
from typing import List, Optional, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.config import OPENAI_API_KEY, LLM_MODEL, TOP_K_RETRIEVAL
from src.document_processor import DocumentProcessor
from src.citations import CitationManager, ConfidenceCalculator, ConfidenceScore
from src.yoy_analysis import YoYAnalyzer
from src.audit_logger import AuditLogger, get_audit_logger


class RAGEngine:
    """RAG Engine for generating 10-K sections."""

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3,
        )
        self.doc_processor = DocumentProcessor()
        self.doc_processor.load_vector_store()
        
        # Enhanced features
        self.citation_manager = CitationManager()
        self.confidence_calculator = ConfidenceCalculator()
        self.yoy_analyzer = YoYAnalyzer()
        self.audit_logger = audit_logger or get_audit_logger()
        
        # Store last generation metadata
        self.last_sources: List[Document] = []
        self.last_confidence: Optional[ConfidenceScore] = None

    def retrieve_context(
        self,
        query: str,
        ticker: str,
        section: Optional[str] = None,
        k: int = TOP_K_RETRIEVAL,
    ) -> List[Document]:
        """Retrieve relevant context from vector store."""
        return self.doc_processor.similarity_search(
            query=query,
            k=k,
            filter_ticker=ticker,
            filter_section=section,
        )

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            meta = doc.metadata
            header = f"[Source {i}: {meta.get('company_name', 'Unknown')} - {meta.get('section', 'Unknown')} - Filed: {meta.get('filing_date', 'Unknown')}]"
            context_parts.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(context_parts)

    def generate_business_section(
        self,
        ticker: str,
        fiscal_year: str,
        additional_context: Optional[str] = None,
        include_citations: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate Item 1 - Business section with citations and confidence.
        
        Returns:
            Tuple of (generated_text, metadata)
        """
        # Retrieve relevant business context
        query = f"company business description operations products services markets for {ticker}"
        docs = self.retrieve_context(query, ticker, section="item_1_business")
        
        if not docs:
            # Fallback to broader search
            docs = self.retrieve_context(query, ticker)
        
        self.last_sources = docs
        
        # Format context with citations
        if include_citations:
            context, citation_map = self.citation_manager.format_citations_for_prompt(docs)
        else:
            context = self.format_context(docs)
            citation_map = {}
        
        prompt = f"""You are a securities lawyer assistant helping to draft SEC Form 10-K filings.

Based on the following context from prior 10-K filings, generate an updated "Item 1. Business" section for {ticker}'s Form 10-K for fiscal year {fiscal_year}.

CONTEXT FROM PRIOR FILINGS:
{context}

{f"ADDITIONAL INFORMATION PROVIDED BY USER:{chr(10)}{additional_context}" if additional_context else ""}

INSTRUCTIONS:
1. Write in the formal, objective tone expected in SEC filings
2. Structure the section with appropriate subsections (e.g., Overview, Products/Services, Markets, Competition, etc.)
3. Base your content on the retrieved context - do NOT hallucinate facts or figures
4. When using information from a specific source, include the source number in brackets, e.g., [Source 1]
5. If you need to reference specific numbers or metrics, clearly indicate they are from prior year filings
6. Keep the narrative factual and compliant with SEC disclosure requirements

Generate the Item 1. Business section:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        generated_text = response.content
        
        # Calculate confidence
        self.last_confidence = self.confidence_calculator.calculate_confidence(
            provided_data={},
            retrieved_docs=docs,
            section="business",
        )
        
        # Log to audit
        self.audit_logger.log_generation(
            section="business",
            generated_text=generated_text,
            sources_used=self.citation_manager.get_citations_json(),
            confidence_score={
                "overall": self.last_confidence.overall,
                "data_coverage": self.last_confidence.data_coverage,
                "source_quality": self.last_confidence.source_quality,
            },
            ticker=ticker,
            fiscal_year=fiscal_year,
        )
        
        # Build metadata
        metadata = {
            "citations": self.citation_manager.get_citations_json(),
            "confidence": {
                "overall": self.last_confidence.overall,
                "data_coverage": self.last_confidence.data_coverage,
                "source_quality": self.last_confidence.source_quality,
                "reasoning": self.last_confidence.reasoning,
            },
            "sources_count": len(docs),
        }
        
        return generated_text, metadata

    def generate_mda_section(
        self,
        ticker: str,
        fiscal_year: str,
        financial_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None,
        include_citations: bool = True,
        include_yoy_analysis: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate Item 7 - MD&A section with citations, confidence, and YoY analysis.
        
        Returns:
            Tuple of (generated_text, metadata)
        """
        # Retrieve relevant MD&A context
        query = f"management discussion analysis financial performance revenue operations results for {ticker}"
        docs = self.retrieve_context(query, ticker, section="item_7_mda")
        
        if not docs:
            docs = self.retrieve_context(query, ticker)
        
        self.last_sources = docs
        
        # Format context with citations
        if include_citations:
            context, citation_map = self.citation_manager.format_citations_for_prompt(docs)
        else:
            context = self.format_context(docs)
            citation_map = {}
        
        # Perform YoY analysis if data provided
        yoy_analysis = ""
        yoy_metrics = []
        if financial_data and include_yoy_analysis:
            # Make a copy to avoid modifying original
            data_copy = dict(financial_data)
            raw_input = data_copy.pop("raw_input", None)
            
            yoy_metrics = self.yoy_analyzer.analyze_data(data_copy)
            if yoy_metrics:
                yoy_analysis = self.yoy_analyzer.format_yoy_table()
                yoy_narrative = self.yoy_analyzer.generate_yoy_narrative()
            
            # Restore raw_input
            if raw_input:
                data_copy["raw_input"] = raw_input
        
        # Format financial data if provided
        financial_section = ""
        if financial_data:
            financial_section = "\nFINANCIAL AND BUSINESS DATA PROVIDED BY USER:\n"
            # Handle raw_input specially
            data_for_prompt = dict(financial_data)
            raw_input = data_for_prompt.pop("raw_input", None)
            
            for key, value in data_for_prompt.items():
                financial_section += f"- {key}: {value}\n"
            
            if raw_input:
                financial_section += f"\nRaw user input (extract any additional relevant data):\n{raw_input}\n"
            
            # Log data provided
            self.audit_logger.log_data_provided(
                raw_input=raw_input or str(financial_data),
                parsed_data=data_for_prompt,
                ticker=ticker,
                fiscal_year=fiscal_year,
            )
        
        prompt = f"""You are a securities lawyer assistant helping to draft SEC Form 10-K filings.

Based on the following context from prior 10-K filings and user-provided financial data, generate an updated "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations" (MD&A) section for {ticker}'s Form 10-K for fiscal year {fiscal_year}.

CONTEXT FROM PRIOR FILINGS:
{context}
{financial_section}
{f"ADDITIONAL CONTEXT:{chr(10)}{additional_context}" if additional_context else ""}

INSTRUCTIONS:
1. Write in the formal, objective tone expected in SEC filings
2. Structure the MD&A with standard subsections:
   - Overview
   - Results of Operations (include segment breakdowns if provided)
   - Liquidity and Capital Resources
   - Critical Accounting Estimates (if applicable)
3. IMPORTANT: Use the user-provided financial data as the primary source for current year figures
4. Compare current year performance to prior year where appropriate
5. When citing information from prior filings, include the source number in brackets, e.g., [Source 1]
6. Explain drivers of performance changes based on business and operational inputs
7. Do NOT hallucinate financial figures - only use what is provided or clearly labeled as prior year data
8. If critical data is missing, note what would typically be included
9. Incorporate any business updates (new products, acquisitions, market expansions) into the narrative
10. Address any operational changes or events mentioned by the user

Generate the Item 7. MD&A section:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        generated_text = response.content
        
        # Calculate confidence
        self.last_confidence = self.confidence_calculator.calculate_confidence(
            provided_data=financial_data or {},
            retrieved_docs=docs,
            section="mda",
        )
        
        # Log to audit
        self.audit_logger.log_generation(
            section="mda",
            generated_text=generated_text,
            sources_used=self.citation_manager.get_citations_json(),
            confidence_score={
                "overall": self.last_confidence.overall,
                "data_coverage": self.last_confidence.data_coverage,
                "source_quality": self.last_confidence.source_quality,
            },
            ticker=ticker,
            fiscal_year=fiscal_year,
        )
        
        # Build metadata
        metadata = {
            "citations": self.citation_manager.get_citations_json(),
            "confidence": {
                "overall": self.last_confidence.overall,
                "data_coverage": self.last_confidence.data_coverage,
                "source_quality": self.last_confidence.source_quality,
                "reasoning": self.last_confidence.reasoning,
            },
            "yoy_analysis": self.yoy_analyzer.get_metrics_json() if yoy_metrics else [],
            "yoy_table": yoy_analysis,
            "sources_count": len(docs),
        }
        
        return generated_text, metadata
    
    def get_citation_references(self) -> str:
        """Get formatted citation references."""
        return self.citation_manager.get_citation_references()
    
    def get_confidence_indicator(self) -> str:
        """Get formatted confidence indicator."""
        if self.last_confidence:
            return self.confidence_calculator.format_confidence_indicator(self.last_confidence)
        return ""
    
    def save_audit_log(self) -> str:
        """Save audit log and return file path."""
        path = self.audit_logger.save_log()
        return str(path)
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit session summary."""
        return self.audit_logger.get_session_summary()

    def update_business_section(
        self,
        ticker: str,
        fiscal_year: str,
        original_section: str,
        business_updates: Dict[str, Any],
    ) -> str:
        """Update Business section with new business information."""
        updates_text = "\nBUSINESS UPDATES PROVIDED BY USER:\n"
        for key, value in business_updates.items():
            updates_text += f"- {key}: {value}\n"
        
        prompt = f"""You are a securities lawyer assistant helping to update SEC Form 10-K filings.

Here is the original Item 1. Business section draft:

{original_section}

The user has provided the following updates for fiscal year {fiscal_year}:
{updates_text}

INSTRUCTIONS:
1. Update the Business section to incorporate the new information
2. Maintain the formal, objective tone expected in SEC filings
3. Keep the same structure but add/modify content as needed
4. Ensure all updates are factually incorporated
5. Do not remove existing accurate information unless contradicted

Generate the updated Item 1. Business section:"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def identify_missing_data(
        self,
        ticker: str,
        section: str = "mda",
    ) -> Dict[str, List[str]]:
        """Identify what financial/business data is needed for complete disclosure."""
        
        missing_data = {
            "financial_inputs": [
                "Total revenue and year-over-year growth rate",
                "Revenue breakdown by segment or product line",
                "Operating income/loss and operating margin",
                "Net income/loss",
                "Adjusted EBITDA (if applicable)",
                "Free cash flow",
                "Cash and cash equivalents balance",
                "Total debt balance",
                "Major capital expenditures",
            ],
            "business_inputs": [
                "New products or services launched in the fiscal year",
                "Products or services discontinued",
                "Market expansions or new geographic entries",
                "Major partnerships or joint ventures",
                "Key acquisitions or divestitures",
            ],
            "operational_inputs": [
                "Changes in pricing or business model",
                "Changes in operational policies",
                "Significant operational events (e.g., outages, incidents)",
                "Regulatory actions or legal proceedings",
                "Key risk factors that emerged during the year",
            ],
        }
        
        return missing_data

    def ask_clarifying_questions(
        self,
        ticker: str,
        fiscal_year: str,
    ) -> str:
        """Generate clarifying questions for missing data."""
        missing = self.identify_missing_data(ticker)
        
        questions = f"""To complete the MD&A section for {ticker}'s Form 10-K for fiscal year {fiscal_year}, I need the following information:

**Financial Data Required:**
"""
        for item in missing["financial_inputs"]:
            questions += f"• {item}\n"
        
        questions += "\n**Business Updates:**\n"
        for item in missing["business_inputs"]:
            questions += f"• {item}\n"
        
        questions += "\n**Operational Information:**\n"
        for item in missing["operational_inputs"]:
            questions += f"• {item}\n"
        
        questions += """
You can provide this information in any format:
- Plain text with numbers
- Markdown table
- HTML table
- Pasted financial statements

Which of these would you like to provide first?"""
        
        return questions

