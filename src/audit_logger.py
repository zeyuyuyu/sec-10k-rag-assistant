"""Audit-friendly logging of user inputs and generated outputs."""
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import uuid

from src.config import DATA_DIR


# Create audit logs directory
AUDIT_DIR = DATA_DIR / "audit_logs"
AUDIT_DIR.mkdir(exist_ok=True)


@dataclass
class AuditEntry:
    """Represents a single audit log entry."""
    timestamp: str
    session_id: str
    event_type: str  # "user_input", "data_provided", "generation", "revision"
    ticker: Optional[str]
    fiscal_year: Optional[str]
    content_hash: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]


class AuditLogger:
    """Audit logger for tracking user inputs and generated outputs."""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.entries: List[AuditEntry] = []
        self.log_file = AUDIT_DIR / f"audit_{self.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def _hash_content(self, content: Any) -> str:
        """Generate hash of content for integrity verification."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def _create_entry(
        self,
        event_type: str,
        content: Dict[str, Any],
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """Create an audit entry."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            session_id=self.session_id,
            event_type=event_type,
            ticker=ticker,
            fiscal_year=fiscal_year,
            content_hash=self._hash_content(content),
            content=content,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        return entry

    def log_user_request(
        self,
        message: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> AuditEntry:
        """Log a user request/message."""
        return self._create_entry(
            event_type="user_request",
            content={"message": message},
            ticker=ticker,
            fiscal_year=fiscal_year,
            metadata={"message_length": len(message)},
        )

    def log_data_provided(
        self,
        raw_input: str,
        parsed_data: Dict[str, Any],
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> AuditEntry:
        """Log user-provided financial data."""
        return self._create_entry(
            event_type="data_provided",
            content={
                "raw_input": raw_input,
                "parsed_data": parsed_data,
            },
            ticker=ticker,
            fiscal_year=fiscal_year,
            metadata={
                "input_length": len(raw_input),
                "fields_parsed": len(parsed_data),
                "fields": list(parsed_data.keys()),
            },
        )

    def log_generation(
        self,
        section: str,  # "business" or "mda"
        generated_text: str,
        sources_used: List[Dict[str, Any]],
        confidence_score: Optional[Dict[str, Any]] = None,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> AuditEntry:
        """Log a generated section."""
        return self._create_entry(
            event_type="generation",
            content={
                "section": section,
                "generated_text": generated_text,
                "text_length": len(generated_text),
            },
            ticker=ticker,
            fiscal_year=fiscal_year,
            metadata={
                "sources_count": len(sources_used),
                "sources": sources_used,
                "confidence": confidence_score,
            },
        )

    def log_revision(
        self,
        section: str,
        original_text: str,
        revised_text: str,
        revision_reason: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[str] = None,
    ) -> AuditEntry:
        """Log a revision to generated content."""
        return self._create_entry(
            event_type="revision",
            content={
                "section": section,
                "original_hash": self._hash_content(original_text),
                "revised_text": revised_text,
                "revision_reason": revision_reason,
            },
            ticker=ticker,
            fiscal_year=fiscal_year,
            metadata={
                "original_length": len(original_text),
                "revised_length": len(revised_text),
                "change_ratio": len(revised_text) / max(len(original_text), 1),
            },
        )

    def save_log(self) -> Path:
        """Save audit log to file."""
        log_data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "total_entries": len(self.entries),
            "entries": [asdict(e) for e in self.entries],
        }
        
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return self.log_file

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the audit session."""
        event_counts = {}
        for entry in self.entries:
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_entries": len(self.entries),
            "event_counts": event_counts,
            "tickers_processed": list(set(e.ticker for e in self.entries if e.ticker)),
            "fiscal_years": list(set(e.fiscal_year for e in self.entries if e.fiscal_year)),
            "log_file": str(self.log_file),
        }

    def generate_audit_report(self) -> str:
        """Generate human-readable audit report."""
        report = f"""
# Audit Report

**Session ID:** {self.session_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Events:** {len(self.entries)}

## Event Timeline

"""
        for entry in self.entries:
            report += f"### {entry.timestamp}\n"
            report += f"**Type:** {entry.event_type}\n"
            if entry.ticker:
                report += f"**Company:** {entry.ticker}\n"
            if entry.fiscal_year:
                report += f"**Fiscal Year:** {entry.fiscal_year}\n"
            report += f"**Content Hash:** `{entry.content_hash}`\n"
            
            if entry.event_type == "data_provided":
                fields = entry.metadata.get("fields", [])
                report += f"**Data Fields Provided:** {', '.join(fields)}\n"
            elif entry.event_type == "generation":
                report += f"**Section:** {entry.content.get('section', 'N/A')}\n"
                report += f"**Text Length:** {entry.content.get('text_length', 0)} characters\n"
                if entry.metadata.get("confidence"):
                    conf = entry.metadata["confidence"]
                    report += f"**Confidence:** {conf.get('overall', 'N/A')}\n"
            
            report += "\n---\n\n"
        
        return report


def get_audit_logger(session_id: Optional[str] = None) -> AuditLogger:
    """Factory function to get audit logger instance."""
    return AuditLogger(session_id)

