"""Source citations and confidence indicators for generated content."""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class Citation:
    """Represents a source citation."""
    source_id: int
    company: str
    section: str
    filing_date: str
    chunk_index: int
    relevance_score: float
    excerpt: str


@dataclass
class ConfidenceScore:
    """Confidence scoring for generated content."""
    overall: float  # 0-1
    data_coverage: float  # How much required data was provided
    source_quality: float  # Quality of retrieved sources
    reasoning: str


class CitationManager:
    """Manages source citations for generated content."""

    def __init__(self):
        self.citations: List[Citation] = []
        self.citation_counter = 0

    def reset(self):
        """Reset citations for new generation."""
        self.citations = []
        self.citation_counter = 0

    def add_citation(
        self,
        document: Document,
        relevance_score: float = 0.0,
    ) -> int:
        """Add a citation and return its ID."""
        self.citation_counter += 1
        meta = document.metadata
        
        citation = Citation(
            source_id=self.citation_counter,
            company=meta.get("company_name", "Unknown"),
            section=meta.get("section", "Unknown"),
            filing_date=meta.get("filing_date", "Unknown"),
            chunk_index=meta.get("chunk_index", 0),
            relevance_score=relevance_score,
            excerpt=document.page_content[:200] + "..." if len(document.page_content) > 200 else document.page_content,
        )
        self.citations.append(citation)
        return self.citation_counter

    def format_citations_for_prompt(self, documents: List[Document]) -> Tuple[str, Dict[int, Citation]]:
        """Format documents with citation markers for LLM prompt."""
        self.reset()
        citation_map = {}
        formatted_parts = []
        
        for doc in documents:
            cite_id = self.add_citation(doc)
            citation_map[cite_id] = self.citations[-1]
            
            meta = doc.metadata
            header = f"[Source {cite_id}] ({meta.get('company_name', 'Unknown')} - {meta.get('section', 'Unknown')} - Filed: {meta.get('filing_date', 'Unknown')})"
            formatted_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(formatted_parts), citation_map

    def get_citation_references(self) -> str:
        """Generate citation references section."""
        if not self.citations:
            return ""
        
        refs = "\n\n---\n\n## Sources\n\n"
        for cite in self.citations:
            refs += f"**[{cite.source_id}]** {cite.company} - {cite.section} (Filed: {cite.filing_date})\n"
        return refs

    def get_citations_json(self) -> List[Dict[str, Any]]:
        """Get citations as JSON-serializable list."""
        return [
            {
                "id": c.source_id,
                "company": c.company,
                "section": c.section,
                "filing_date": c.filing_date,
                "chunk_index": c.chunk_index,
                "relevance_score": c.relevance_score,
                "excerpt": c.excerpt,
            }
            for c in self.citations
        ]


class ConfidenceCalculator:
    """Calculates confidence scores for generated content."""

    # Required data fields for MD&A
    REQUIRED_FINANCIAL_FIELDS = [
        "revenue", "growth", "operating income", "net income",
        "cash flow", "ebitda", "margin"
    ]
    
    REQUIRED_BUSINESS_FIELDS = [
        "products", "services", "markets", "acquisitions", "partnerships"
    ]

    def calculate_confidence(
        self,
        provided_data: Dict[str, Any],
        retrieved_docs: List[Document],
        section: str = "mda",
    ) -> ConfidenceScore:
        """Calculate confidence score for generated content."""
        # Calculate data coverage
        data_coverage = self._calculate_data_coverage(provided_data, section)
        
        # Calculate source quality
        source_quality = self._calculate_source_quality(retrieved_docs)
        
        # Overall confidence
        overall = (data_coverage * 0.6 + source_quality * 0.4)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(data_coverage, source_quality, provided_data)
        
        return ConfidenceScore(
            overall=round(overall, 2),
            data_coverage=round(data_coverage, 2),
            source_quality=round(source_quality, 2),
            reasoning=reasoning,
        )

    def _calculate_data_coverage(self, data: Dict[str, Any], section: str) -> float:
        """Calculate how much required data was provided."""
        if not data:
            return 0.3  # Base score for RAG-only generation
        
        data_str = " ".join(str(v).lower() for v in data.values())
        
        if section == "mda":
            required = self.REQUIRED_FINANCIAL_FIELDS
        else:
            required = self.REQUIRED_BUSINESS_FIELDS
        
        matched = sum(1 for field in required if field in data_str)
        coverage = matched / len(required)
        
        # Boost if raw_input provided (more context)
        if "raw_input" in data:
            coverage = min(1.0, coverage + 0.2)
        
        return min(1.0, coverage + 0.3)  # Base 0.3 for having some data

    def _calculate_source_quality(self, docs: List[Document]) -> float:
        """Calculate quality of retrieved sources."""
        if not docs:
            return 0.0
        
        # More sources = higher quality (up to a point)
        quantity_score = min(1.0, len(docs) / 8)
        
        # Check section diversity
        sections = set(d.metadata.get("section", "") for d in docs)
        diversity_score = min(1.0, len(sections) / 3)
        
        # Check recency (prefer recent filings)
        recency_scores = []
        for doc in docs:
            filing_date = doc.metadata.get("filing_date", "")
            if "2024" in filing_date or "2025" in filing_date:
                recency_scores.append(1.0)
            elif "2023" in filing_date:
                recency_scores.append(0.8)
            else:
                recency_scores.append(0.5)
        
        recency_score = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        
        return (quantity_score * 0.3 + diversity_score * 0.3 + recency_score * 0.4)

    def _generate_reasoning(
        self,
        data_coverage: float,
        source_quality: float,
        provided_data: Dict[str, Any],
    ) -> str:
        """Generate human-readable confidence reasoning."""
        reasons = []
        
        if data_coverage >= 0.8:
            reasons.append("Comprehensive financial data provided")
        elif data_coverage >= 0.5:
            reasons.append("Partial financial data provided")
        else:
            reasons.append("Limited financial data - some sections may be incomplete")
        
        if source_quality >= 0.8:
            reasons.append("Strong source coverage from prior filings")
        elif source_quality >= 0.5:
            reasons.append("Adequate source coverage")
        else:
            reasons.append("Limited source material available")
        
        # Check for specific data
        if provided_data:
            data_str = str(provided_data).lower()
            if "revenue" in data_str:
                reasons.append("Revenue figures grounded in provided data")
            if "growth" in data_str:
                reasons.append("Growth metrics available for comparison")
        
        return "; ".join(reasons)

    def format_confidence_indicator(self, score: ConfidenceScore) -> str:
        """Format confidence score as human-readable indicator."""
        if score.overall >= 0.8:
            level = "HIGH"
            emoji = "🟢"
        elif score.overall >= 0.6:
            level = "MEDIUM"
            emoji = "🟡"
        else:
            level = "LOW"
            emoji = "🔴"
        
        return f"""
---

**Confidence Assessment** {emoji}

| Metric | Score |
|--------|-------|
| Overall Confidence | {level} ({score.overall:.0%}) |
| Data Coverage | {score.data_coverage:.0%} |
| Source Quality | {score.source_quality:.0%} |

*{score.reasoning}*

---
"""

