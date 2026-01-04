"""Interactive 10-K Assistant with conversation management."""
import re
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.config import OPENAI_API_KEY, LLM_MODEL, TARGET_COMPANIES
from src.rag_engine import RAGEngine


class ConversationState(Enum):
    """States for the conversation flow."""
    INITIAL = "initial"
    AWAITING_COMPANY = "awaiting_company"
    AWAITING_YEAR = "awaiting_year"
    AWAITING_FINANCIAL_DATA = "awaiting_financial_data"
    GENERATING_BUSINESS = "generating_business"
    GENERATING_MDA = "generating_mda"
    COMPLETE = "complete"


@dataclass
class ConversationContext:
    """Holds the conversation context and collected data."""
    state: ConversationState = ConversationState.INITIAL
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    fiscal_year: Optional[str] = None
    financial_data: Dict[str, Any] = field(default_factory=dict)
    business_inputs: Dict[str, Any] = field(default_factory=dict)
    operational_inputs: Dict[str, Any] = field(default_factory=dict)
    generated_sections: Dict[str, str] = field(default_factory=dict)
    messages: List[Dict[str, str]] = field(default_factory=list)


class TenKAssistant:
    """Interactive assistant for 10-K generation."""

    SYSTEM_PROMPT = """You are a helpful legal assistant specializing in SEC Form 10-K filings. 
You help securities lawyers and finance professionals draft Business and MD&A sections.

Your communication style should be:
- Clear and professional
- Business-friendly (avoid technical AI/ML jargon)
- Patient and helpful like a human legal assistant
- Structured and organized in your questions

When collecting financial data:
- Ask clear, specific questions
- Accept data in any format (tables, plain text, pasted statements)
- Confirm understanding of provided data
- Be helpful in organizing the information

Available companies for 10-K generation:
NVDA (NVIDIA), MSFT (Microsoft), KO (Coca-Cola), NKE (Nike), 
AMZN (Amazon), DASH (DoorDash), TJX (TJX Companies), DRI (Darden Restaurants)"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.7,
        )
        self.rag_engine = RAGEngine()
        self.context = ConversationContext()

    def reset(self):
        """Reset the conversation context."""
        self.context = ConversationContext()

    def _add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.context.messages.append({"role": role, "content": content})

    def _parse_ticker(self, text: str) -> Optional[str]:
        """Extract ticker from user input."""
        text_upper = text.upper()
        for ticker in TARGET_COMPANIES:
            if ticker in text_upper:
                return ticker
            if TARGET_COMPANIES[ticker]["name"].upper() in text_upper:
                return ticker
        return None

    def _parse_year(self, text: str) -> Optional[str]:
        """Extract fiscal year from user input."""
        # Look for 4-digit year
        match = re.search(r'20\d{2}', text)
        if match:
            return match.group()
        return None

    def _parse_html_table(self, html: str) -> Dict[str, Any]:
        """Parse HTML table to extract financial data."""
        from bs4 import BeautifulSoup
        data = {}
        
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            headers = []
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                cell_texts = [c.get_text(strip=True) for c in cells]
                
                if not headers:
                    # First row with content becomes headers
                    if any(cell_texts):
                        headers = cell_texts
                elif len(cell_texts) >= 2:
                    metric = cell_texts[0]
                    values = cell_texts[1:]
                    if metric and values[0]:
                        data[metric] = values[0]
                        if len(values) >= 2 and values[1]:
                            data[f"{metric} (Prior Year)"] = values[1]
        
        return data

    def _parse_financial_data(self, text: str) -> Dict[str, Any]:
        """Parse financial data from user input (supports Markdown, HTML, plain text)."""
        data = {}
        
        # Try to parse HTML table first
        if '<table' in text.lower() or '<tr' in text.lower():
            html_data = self._parse_html_table(text)
            data.update(html_data)
        
        # Try to parse markdown table
        if '|' in text:
            lines = text.strip().split('\n')
            headers = []
            for line in lines:
                if '|' in line and '---' not in line:
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if not headers:
                        headers = cells
                    elif len(cells) >= 2:
                        metric = cells[0]
                        values = cells[1:]
                        if len(values) >= 1:
                            data[metric] = values[0]
                        if len(values) >= 2:
                            data[f"{metric} (Prior Year)"] = values[1]
        
        # Parse key-value patterns (plain text)
        patterns = [
            r'(?i)(revenue|sales)[\s:]*\$?([\d,.]+)\s*(billion|million|B|M)?',
            r'(?i)(growth|increase|decrease)[\s:]*(-?[\d,.]+)%?',
            r'(?i)(operating income|net income|EBITDA)[\s:]*\$?([\d,.]+)\s*(billion|million|B|M)?',
            r'(?i)(cash flow|FCF|free cash flow)[\s:]*\$?([\d,.]+)\s*(billion|million|B|M)?',
            r'(?i)(margin)[\s:]*(-?[\d,.]+)%',
            r'(?i)(segment|division)[\s:]+([^\n,]+)',
            r'(?i)(launched|discontinued|acquired|partnered)[\s:]+([^\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                key = match[0].strip()
                value = match[1].strip()
                unit = match[2].strip() if len(match) > 2 else ""
                if unit:
                    value = f"{value} {unit}"
                if key not in data:
                    data[key] = value
        
        # Also store raw input for LLM to process
        if text.strip():
            data["raw_input"] = text.strip()
        
        return data

    def _get_initial_response(self) -> str:
        """Generate initial greeting."""
        return """Hello! I'm your 10-K filing assistant. I can help you draft the Business (Item 1) and MD&A (Item 7) sections for SEC Form 10-K filings.

I have access to prior year 10-K filings for these companies:
• **NVDA** - NVIDIA Corporation
• **MSFT** - Microsoft Corporation  
• **KO** - The Coca-Cola Company
• **NKE** - NIKE, Inc.
• **AMZN** - Amazon.com, Inc.
• **DASH** - DoorDash, Inc.
• **TJX** - The TJX Companies, Inc.
• **DRI** - Darden Restaurants, Inc.

**How to get started:**
Simply tell me which company's 10-K you'd like to work on, for example:
- "Generate the Business and MD&A sections for NVIDIA's 2024 Form 10-K"
- "Help me draft the 10-K for Microsoft"

Which company would you like to start with?"""

    def _ask_for_company(self) -> str:
        """Ask user to specify company."""
        return """I'd be happy to help! Which company's 10-K filing would you like to work on?

Available companies:
• NVDA (NVIDIA)
• MSFT (Microsoft)
• KO (Coca-Cola)
• NKE (Nike)
• AMZN (Amazon)
• DASH (DoorDash)
• TJX (TJX Companies)
• DRI (Darden Restaurants)

Please specify the company ticker or name."""

    def _ask_for_year(self) -> str:
        """Ask user to specify fiscal year."""
        return f"""Great! I'll help you with {self.context.company_name}'s Form 10-K.

What fiscal year would you like to generate the filing for? 
(For example: 2024, 2023)"""

    def _ask_for_financial_data(self) -> str:
        """Ask for required financial data."""
        return self.rag_engine.ask_clarifying_questions(
            self.context.ticker,
            self.context.fiscal_year,
        )

    def process_message(self, user_message: str) -> str:
        """Process user message and generate response."""
        self._add_message("user", user_message)
        
        # State machine logic
        if self.context.state == ConversationState.INITIAL:
            # Try to extract company and year from initial message
            ticker = self._parse_ticker(user_message)
            year = self._parse_year(user_message)
            
            if ticker:
                self.context.ticker = ticker
                self.context.company_name = TARGET_COMPANIES[ticker]["name"]
                
                if year:
                    self.context.fiscal_year = year
                    self.context.state = ConversationState.GENERATING_BUSINESS
                    response = self._generate_and_ask_financial()
                else:
                    self.context.state = ConversationState.AWAITING_YEAR
                    response = self._ask_for_year()
            else:
                self.context.state = ConversationState.AWAITING_COMPANY
                response = self._ask_for_company()
        
        elif self.context.state == ConversationState.AWAITING_COMPANY:
            ticker = self._parse_ticker(user_message)
            if ticker:
                self.context.ticker = ticker
                self.context.company_name = TARGET_COMPANIES[ticker]["name"]
                self.context.state = ConversationState.AWAITING_YEAR
                response = self._ask_for_year()
            else:
                response = "I couldn't identify that company. Please specify one of the available companies: NVDA, MSFT, KO, NKE, AMZN, DASH, TJX, or DRI."
        
        elif self.context.state == ConversationState.AWAITING_YEAR:
            year = self._parse_year(user_message)
            if year:
                self.context.fiscal_year = year
                self.context.state = ConversationState.GENERATING_BUSINESS
                response = self._generate_and_ask_financial()
            else:
                response = "Please specify a fiscal year (e.g., 2024, 2023)."
        
        elif self.context.state == ConversationState.AWAITING_FINANCIAL_DATA:
            # Parse financial data from user input
            parsed_data = self._parse_financial_data(user_message)
            self.context.financial_data.update(parsed_data)
            
            # Generate MD&A with provided data
            response = self._generate_mda_section()
        
        else:
            # Handle follow-up questions or new requests
            response = self._handle_general_query(user_message)
        
        self._add_message("assistant", response)
        return response

    def _generate_and_ask_financial(self) -> str:
        """Generate Business section and ask for financial data."""
        response = f"""Excellent! I'll generate the 10-K sections for **{self.context.company_name} ({self.context.ticker})** for fiscal year **{self.context.fiscal_year}**.

Let me first retrieve information from prior filings and generate the Business section...

---

## Item 1. Business (Draft)

"""
        # Generate business section with enhanced features
        try:
            business_section, metadata = self.rag_engine.generate_business_section(
                self.context.ticker,
                self.context.fiscal_year,
                include_citations=True,
            )
            self.context.generated_sections["business"] = business_section
            self.context.generated_sections["business_metadata"] = metadata
            response += business_section
            
            # Add citation references
            response += self.rag_engine.get_citation_references()
            
            # Add confidence indicator
            response += self.rag_engine.get_confidence_indicator()
            
        except Exception as e:
            response += f"*[Note: Unable to generate Business section from prior filings. Error: {str(e)}. Please ensure the 10-K filing has been downloaded and indexed.]*"
        
        response += """

---

Now, to generate the **MD&A (Item 7)** section, I need current fiscal year financial and business data.

"""
        response += self._ask_for_financial_data()
        
        self.context.state = ConversationState.AWAITING_FINANCIAL_DATA
        return response

    def _generate_mda_section(self) -> str:
        """Generate MD&A section with provided data."""
        response = f"""Thank you for providing that information! Let me generate the MD&A section incorporating your data...

---

## Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations (Draft)

"""
        try:
            mda_section, metadata = self.rag_engine.generate_mda_section(
                self.context.ticker,
                self.context.fiscal_year,
                self.context.financial_data,
                include_citations=True,
                include_yoy_analysis=True,
            )
            self.context.generated_sections["mda"] = mda_section
            self.context.generated_sections["mda_metadata"] = metadata
            response += mda_section
            
            # Add YoY analysis table if available
            if metadata.get("yoy_table"):
                response += metadata["yoy_table"]
            
            # Add citation references
            response += self.rag_engine.get_citation_references()
            
            # Add confidence indicator
            response += self.rag_engine.get_confidence_indicator()
            
            # Save audit log
            audit_file = self.rag_engine.save_audit_log()
            response += f"\n*Audit log saved to: `{audit_file}`*\n"
            
        except Exception as e:
            response += f"*[Error generating MD&A: {str(e)}]*"
        
        response += """

---

The draft sections are now complete. Would you like me to:
1. **Revise** any specific part of the Business or MD&A sections?
2. **Add more detail** to any subsection?
3. **Incorporate additional data** you'd like to provide?
4. **Generate for a different company**?

Please let me know how I can help further."""
        
        self.context.state = ConversationState.COMPLETE
        return response

    def _handle_general_query(self, query: str) -> str:
        """Handle general queries or follow-up requests."""
        query_lower = query.lower()
        
        # Check if starting new generation
        if any(word in query_lower for word in ["generate", "create", "draft", "new"]):
            ticker = self._parse_ticker(query)
            if ticker:
                self.reset()
                self.context.ticker = ticker
                self.context.company_name = TARGET_COMPANIES[ticker]["name"]
                year = self._parse_year(query)
                if year:
                    self.context.fiscal_year = year
                    self.context.state = ConversationState.GENERATING_BUSINESS
                    return self._generate_and_ask_financial()
                else:
                    self.context.state = ConversationState.AWAITING_YEAR
                    return self._ask_for_year()
        
        # Check if providing more data
        if self.context.state == ConversationState.COMPLETE:
            if any(word in query_lower for word in ["revise", "update", "change", "add"]):
                # Parse any additional data
                parsed_data = self._parse_financial_data(query)
                if parsed_data:
                    self.context.financial_data.update(parsed_data)
                    return self._generate_mda_section()
        
        # Default: conversational response
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"""The user is working on a 10-K for {self.context.ticker or 'unspecified company'}.
            
Current state: {self.context.state.value}
Already generated: {list(self.context.generated_sections.keys())}

User's message: {query}

Respond helpfully as a legal assistant would.""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def get_generated_content(self) -> Dict[str, str]:
        """Return all generated sections."""
        return self.context.generated_sections


def create_assistant() -> TenKAssistant:
    """Factory function to create assistant instance."""
    return TenKAssistant()

