"""Year-over-year comparison and analysis logic."""
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class YoYMetric:
    """Represents a year-over-year metric comparison."""
    name: str
    current_value: float
    prior_value: Optional[float]
    unit: str  # e.g., "B", "M", "%"
    change_absolute: Optional[float]
    change_percent: Optional[float]
    trend: str  # "up", "down", "flat"


class YoYAnalyzer:
    """Analyzes year-over-year changes in financial data."""

    def __init__(self):
        self.metrics: List[YoYMetric] = []

    def parse_value(self, value_str: str) -> Tuple[float, str]:
        """Parse a financial value string into number and unit."""
        if not value_str:
            return 0.0, ""
        
        value_str = str(value_str).strip()
        
        # Remove currency symbols
        value_str = value_str.replace("$", "").replace(",", "")
        
        # Extract unit
        unit = ""
        if "billion" in value_str.lower() or value_str.endswith("B"):
            unit = "B"
            value_str = re.sub(r'[Bb]illion|[Bb]$', '', value_str)
        elif "million" in value_str.lower() or value_str.endswith("M"):
            unit = "M"
            value_str = re.sub(r'[Mm]illion|[Mm]$', '', value_str)
        elif "%" in value_str:
            unit = "%"
            value_str = value_str.replace("%", "")
        
        # Parse number
        try:
            # Handle negative values in parentheses
            if "(" in value_str and ")" in value_str:
                value_str = "-" + value_str.replace("(", "").replace(")", "")
            value = float(value_str.strip())
        except ValueError:
            value = 0.0
        
        return value, unit

    def analyze_data(self, financial_data: Dict[str, Any]) -> List[YoYMetric]:
        """Analyze financial data for year-over-year changes."""
        self.metrics = []
        
        # Group current and prior year data
        current_data = {}
        prior_data = {}
        
        for key, value in financial_data.items():
            if key == "raw_input":
                continue
            
            if "(Prior Year)" in key:
                base_key = key.replace(" (Prior Year)", "")
                prior_data[base_key] = value
            else:
                current_data[key] = value
        
        # Calculate YoY for each metric
        for metric_name, current_value_str in current_data.items():
            current_val, unit = self.parse_value(current_value_str)
            
            prior_val = None
            change_abs = None
            change_pct = None
            trend = "flat"
            
            if metric_name in prior_data:
                prior_val, _ = self.parse_value(prior_data[metric_name])
                
                if prior_val != 0:
                    change_abs = current_val - prior_val
                    change_pct = ((current_val - prior_val) / abs(prior_val)) * 100
                    
                    if change_pct > 1:
                        trend = "up"
                    elif change_pct < -1:
                        trend = "down"
            
            self.metrics.append(YoYMetric(
                name=metric_name,
                current_value=current_val,
                prior_value=prior_val,
                unit=unit,
                change_absolute=round(change_abs, 2) if change_abs is not None else None,
                change_percent=round(change_pct, 1) if change_pct is not None else None,
                trend=trend,
            ))
        
        return self.metrics

    def format_yoy_table(self) -> str:
        """Format YoY analysis as a markdown table."""
        if not self.metrics:
            return ""
        
        table = "\n### Year-over-Year Analysis\n\n"
        table += "| Metric | Current Year | Prior Year | Change | % Change | Trend |\n"
        table += "|--------|--------------|------------|--------|----------|-------|\n"
        
        for m in self.metrics:
            current = f"{m.current_value:.1f}{m.unit}" if m.unit != "%" else f"{m.current_value:.1f}%"
            prior = f"{m.prior_value:.1f}{m.unit}" if m.prior_value is not None else "N/A"
            if m.unit == "%" and m.prior_value is not None:
                prior = f"{m.prior_value:.1f}%"
            
            change = f"{m.change_absolute:+.1f}{m.unit}" if m.change_absolute is not None else "N/A"
            if m.unit == "%":
                change = f"{m.change_absolute:+.1f}pp" if m.change_absolute is not None else "N/A"
            
            pct_change = f"{m.change_percent:+.1f}%" if m.change_percent is not None else "N/A"
            
            trend_icon = {"up": "📈", "down": "📉", "flat": "➡️"}.get(m.trend, "")
            
            table += f"| {m.name} | {current} | {prior} | {change} | {pct_change} | {trend_icon} |\n"
        
        return table

    def generate_yoy_narrative(self) -> str:
        """Generate narrative description of YoY changes."""
        if not self.metrics:
            return ""
        
        narratives = []
        
        # Find significant changes
        significant_up = [m for m in self.metrics if m.change_percent and m.change_percent > 10]
        significant_down = [m for m in self.metrics if m.change_percent and m.change_percent < -10]
        
        if significant_up:
            items = [f"{m.name} (+{m.change_percent:.1f}%)" for m in significant_up[:3]]
            narratives.append(f"Notable increases: {', '.join(items)}")
        
        if significant_down:
            items = [f"{m.name} ({m.change_percent:.1f}%)" for m in significant_down[:3]]
            narratives.append(f"Notable decreases: {', '.join(items)}")
        
        # Revenue specific analysis
        revenue_metrics = [m for m in self.metrics if "revenue" in m.name.lower()]
        if revenue_metrics:
            rev = revenue_metrics[0]
            if rev.change_percent:
                if rev.change_percent > 20:
                    narratives.append(f"Strong revenue growth of {rev.change_percent:.1f}% year-over-year")
                elif rev.change_percent > 0:
                    narratives.append(f"Revenue increased {rev.change_percent:.1f}% compared to prior year")
                else:
                    narratives.append(f"Revenue declined {abs(rev.change_percent):.1f}% from prior year")
        
        return "; ".join(narratives) if narratives else "Year-over-year data available for comparison"

    def get_metrics_json(self) -> List[Dict[str, Any]]:
        """Get metrics as JSON-serializable list."""
        return [
            {
                "name": m.name,
                "current_value": m.current_value,
                "prior_value": m.prior_value,
                "unit": m.unit,
                "change_absolute": m.change_absolute,
                "change_percent": m.change_percent,
                "trend": m.trend,
            }
            for m in self.metrics
        ]

