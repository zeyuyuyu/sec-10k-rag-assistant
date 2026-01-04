"""Document processing and vectorization for 10-K filings."""
import json
import re
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import (
    FILINGS_DIR,
    VECTOR_DB_DIR,
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


class DocumentProcessor:
    """Processes 10-K filings and creates vector store."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        self.vector_store: Optional[FAISS] = None

    def load_filing(self, ticker: str) -> Optional[dict]:
        """Load a 10-K filing from disk."""
        filing_path = FILINGS_DIR / f"{ticker}_10k.json"
        if not filing_path.exists():
            print(f"Filing not found: {filing_path}")
            return None
        
        with open(filing_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_documents_from_filing(self, filing_data: dict) -> List[Document]:
        """Create LangChain documents from a 10-K filing."""
        documents = []
        ticker = filing_data["ticker"]
        company_name = filing_data["company_name"]
        filing_date = filing_data["filing_date"]
        
        sections = filing_data.get("sections", {})
        
        # Process each section
        section_mappings = {
            "item_1_business": "Item 1 - Business",
            "item_1a_risk_factors": "Item 1A - Risk Factors",
            "item_7_mda": "Item 7 - MD&A",
            "item_7a_market_risk": "Item 7A - Market Risk",
        }
        
        for section_key, section_name in section_mappings.items():
            content = sections.get(section_key, "")
            if not content or len(content) < 100:
                continue
            
            # Clean the content
            content = self._clean_text(content)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "ticker": ticker,
                        "company_name": company_name,
                        "filing_date": filing_date,
                        "section": section_name,
                        "section_key": section_key,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
                documents.append(doc)
        
        return documents

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'Table of Contents', '', text, flags=re.IGNORECASE)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        return text.strip()

    def build_vector_store(self, tickers: Optional[List[str]] = None) -> FAISS:
        """Build vector store from 10-K filings."""
        all_documents = []
        
        # Get list of tickers to process
        if tickers is None:
            filing_files = list(FILINGS_DIR.glob("*_10k.json"))
            tickers = [f.stem.replace("_10k", "").upper() for f in filing_files]
        
        print(f"Processing filings for: {tickers}")
        
        for ticker in tickers:
            filing_data = self.load_filing(ticker)
            if filing_data:
                docs = self.create_documents_from_filing(filing_data)
                all_documents.extend(docs)
                print(f"  {ticker}: {len(docs)} chunks")
        
        if not all_documents:
            raise ValueError("No documents to process")
        
        print(f"\nTotal documents: {len(all_documents)}")
        print("Creating vector store...")
        
        self.vector_store = FAISS.from_documents(
            all_documents,
            self.embeddings,
        )
        
        # Save vector store
        self.save_vector_store()
        
        return self.vector_store

    def save_vector_store(self) -> None:
        """Save vector store to disk."""
        if self.vector_store:
            self.vector_store.save_local(str(VECTOR_DB_DIR / "faiss_index"))
            print(f"Vector store saved to {VECTOR_DB_DIR / 'faiss_index'}")

    def load_vector_store(self) -> Optional[FAISS]:
        """Load vector store from disk."""
        index_path = VECTOR_DB_DIR / "faiss_index"
        if index_path.exists():
            self.vector_store = FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Vector store loaded successfully")
            return self.vector_store
        return None

    def get_or_create_vector_store(self, tickers: Optional[List[str]] = None) -> FAISS:
        """Get existing vector store or create new one."""
        # Try to load existing
        store = self.load_vector_store()
        if store:
            return store
        
        # Create new one
        return self.build_vector_store(tickers)

    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_ticker: Optional[str] = None,
        filter_section: Optional[str] = None,
    ) -> List[Document]:
        """Search for relevant documents."""
        if not self.vector_store:
            self.load_vector_store()
        
        if not self.vector_store:
            raise ValueError("No vector store available")
        
        # Build filter
        filter_dict = {}
        if filter_ticker:
            filter_dict["ticker"] = filter_ticker
        if filter_section:
            filter_dict["section_key"] = filter_section
        
        if filter_dict:
            results = self.vector_store.similarity_search(
                query, 
                k=k,
                filter=filter_dict,
            )
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        return results


def main():
    """Build vector store from downloaded filings."""
    processor = DocumentProcessor()
    processor.build_vector_store()


if __name__ == "__main__":
    main()

