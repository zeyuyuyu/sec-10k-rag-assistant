"""SEC EDGAR 10-K Filing Downloader."""
import re
import time
import json
import requests
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
import html2text

from src.config import (
    SEC_BASE_URL,
    USER_AGENT,
    TARGET_COMPANIES,
    FILINGS_DIR,
)


class SECDownloader:
    """Downloads and processes SEC 10-K filings from EDGAR."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = False
        self.h2t.ignore_images = True
        self.h2t.body_width = 0

    def get_company_filings(self, cik: str) -> dict:
        """Get recent filings for a company by CIK."""
        # Remove leading zeros for API call
        cik_clean = cik.lstrip("0")
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching filings for CIK {cik}: {e}")
            return {}

    def find_10k_filing(self, filings_data: dict) -> Optional[dict]:
        """Find the most recent 10-K filing from company filings data."""
        if not filings_data or "filings" not in filings_data:
            return None
        
        recent = filings_data["filings"].get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_documents = recent.get("primaryDocument", [])
        
        for i, form in enumerate(forms):
            if form == "10-K":
                return {
                    "form": form,
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_documents[i],
                }
        return None

    def download_10k_html(self, cik: str, accession_number: str, primary_doc: str) -> Optional[str]:
        """Download the 10-K HTML filing."""
        # Format accession number (remove dashes)
        accession_clean = accession_number.replace("-", "")
        url = f"{SEC_BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/{accession_clean}/{primary_doc}"
        
        try:
            time.sleep(0.1)  # SEC rate limiting
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error downloading 10-K: {e}")
            return None

    def parse_10k_sections(self, html_content: str) -> dict:
        """Parse 10-K HTML to extract key sections."""
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()
        
        # Get text content
        text = soup.get_text(separator="\n", strip=True)
        
        # Also get structured HTML sections
        sections = {
            "full_text": text,
            "item_1_business": self._extract_section(text, "Item 1", "Item 1A"),
            "item_1a_risk_factors": self._extract_section(text, "Item 1A", "Item 1B"),
            "item_7_mda": self._extract_section(text, "Item 7", "Item 7A"),
            "item_7a_market_risk": self._extract_section(text, "Item 7A", "Item 8"),
        }
        
        return sections

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract a section between two markers."""
        # Common patterns for section headers in 10-K
        patterns = [
            rf"(?i){start_marker}[\.\s]*[-–—]?\s*(Business|Risk Factors|Management'?s? Discussion|Quantitative)",
            rf"(?i)ITEM\s*{start_marker[-1]}[A]?[\.\s]*[-–—]?",
        ]
        
        start_idx = -1
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                start_idx = match.start()
                break
        
        if start_idx == -1:
            # Try simpler pattern
            simple_match = re.search(rf"(?i){start_marker}", text)
            if simple_match:
                start_idx = simple_match.start()
        
        if start_idx == -1:
            return ""
        
        # Find end marker
        end_patterns = [
            rf"(?i){end_marker}[\.\s]*[-–—]?",
            rf"(?i)ITEM\s*{end_marker[-2:]}[\.\s]*[-–—]?",
        ]
        
        end_idx = len(text)
        search_text = text[start_idx + 100:]  # Skip past start marker
        
        for pattern in end_patterns:
            match = re.search(pattern, search_text)
            if match:
                end_idx = start_idx + 100 + match.start()
                break
        
        extracted = text[start_idx:end_idx]
        
        # Limit section length
        if len(extracted) > 200000:
            extracted = extracted[:200000]
        
        return extracted.strip()

    def download_company_10k(self, ticker: str) -> Optional[dict]:
        """Download and process 10-K for a specific company."""
        if ticker not in TARGET_COMPANIES:
            print(f"Unknown ticker: {ticker}")
            return None
        
        company_info = TARGET_COMPANIES[ticker]
        cik = company_info["cik"]
        
        print(f"Fetching filings for {company_info['name']} ({ticker})...")
        filings_data = self.get_company_filings(cik)
        
        if not filings_data:
            return None
        
        filing_info = self.find_10k_filing(filings_data)
        if not filing_info:
            print(f"No 10-K filing found for {ticker}")
            return None
        
        print(f"Found 10-K filed on {filing_info['filing_date']}")
        
        html_content = self.download_10k_html(
            cik, 
            filing_info["accession_number"],
            filing_info["primary_document"]
        )
        
        if not html_content:
            return None
        
        print("Parsing 10-K sections...")
        sections = self.parse_10k_sections(html_content)
        
        # Save to file
        output_file = FILINGS_DIR / f"{ticker}_10k.json"
        result = {
            "ticker": ticker,
            "company_name": company_info["name"],
            "cik": cik,
            "filing_date": filing_info["filing_date"],
            "accession_number": filing_info["accession_number"],
            "sections": sections,
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Saved to {output_file}")
        return result

    def download_all_companies(self) -> list:
        """Download 10-K filings for all target companies."""
        results = []
        for ticker in TARGET_COMPANIES:
            try:
                result = self.download_company_10k(ticker)
                if result:
                    results.append(result)
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        return results


def main():
    """Download all 10-K filings."""
    downloader = SECDownloader()
    downloader.download_all_companies()


if __name__ == "__main__":
    main()

