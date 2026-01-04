"""Command-line interface for 10-K RAG Assistant."""
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.assistant import create_assistant, TenKAssistant
from src.sec_downloader import SECDownloader
from src.document_processor import DocumentProcessor
from src.config import TARGET_COMPANIES

app = typer.Typer(help="SEC 10-K RAG Assistant CLI")
console = Console()


@app.command()
def chat():
    """Start an interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]SEC 10-K RAG Assistant[/bold blue]\n"
        "Interactive mode for drafting Form 10-K sections\n"
        "[dim]Type 'quit' or 'exit' to end the session[/dim]",
        border_style="blue"
    ))
    
    assistant = create_assistant()
    
    # Show initial greeting
    initial_response = assistant._get_initial_response()
    assistant._add_message("assistant", initial_response)
    console.print()
    console.print(Markdown(initial_response))
    console.print()
    
    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")
            
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[dim]Goodbye![/dim]")
                break
            
            if not user_input.strip():
                continue
            
            # Process message
            response = assistant.process_message(user_input)
            
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold blue]Assistant[/bold blue]",
                border_style="blue",
            ))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[dim]Session ended.[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def download(
    ticker: Optional[str] = typer.Argument(None, help="Company ticker (e.g., NVDA)"),
    all_companies: bool = typer.Option(False, "--all", "-a", help="Download all companies"),
):
    """Download 10-K filings from SEC EDGAR."""
    downloader = SECDownloader()
    
    if all_companies:
        console.print("[bold]Downloading 10-K filings for all companies...[/bold]")
        results = downloader.download_all_companies()
        console.print(f"[green]Downloaded {len(results)} filings[/green]")
    elif ticker:
        ticker = ticker.upper()
        if ticker not in TARGET_COMPANIES:
            console.print(f"[red]Unknown ticker: {ticker}[/red]")
            console.print(f"Available: {', '.join(TARGET_COMPANIES.keys())}")
            raise typer.Exit(1)
        
        console.print(f"[bold]Downloading 10-K for {ticker}...[/bold]")
        result = downloader.download_company_10k(ticker)
        if result:
            console.print(f"[green]Successfully downloaded {ticker} 10-K[/green]")
        else:
            console.print(f"[red]Failed to download {ticker} 10-K[/red]")
    else:
        console.print("[yellow]Please specify a ticker or use --all flag[/yellow]")
        console.print(f"Available tickers: {', '.join(TARGET_COMPANIES.keys())}")


@app.command()
def index(
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Rebuild vector store from scratch"),
):
    """Build or update the vector index."""
    processor = DocumentProcessor()
    
    if rebuild:
        console.print("[bold]Rebuilding vector store...[/bold]")
    else:
        console.print("[bold]Building/updating vector store...[/bold]")
    
    try:
        if rebuild:
            processor.build_vector_store()
        else:
            processor.get_or_create_vector_store()
        console.print("[green]Vector store ready![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate(
    ticker: str = typer.Argument(..., help="Company ticker (e.g., NVDA)"),
    year: str = typer.Argument(..., help="Fiscal year (e.g., 2024)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """Generate 10-K sections (Business only, without financial data)."""
    ticker = ticker.upper()
    
    if ticker not in TARGET_COMPANIES:
        console.print(f"[red]Unknown ticker: {ticker}[/red]")
        raise typer.Exit(1)
    
    assistant = create_assistant()
    
    console.print(f"[bold]Generating Business section for {ticker} FY{year}...[/bold]")
    
    try:
        business_section = assistant.rag_engine.generate_business_section(ticker, year)
        
        output_text = f"""# {TARGET_COMPANIES[ticker]['name']} ({ticker})
# Form 10-K - Fiscal Year {year}

## Item 1. Business

{business_section}

---
*Note: MD&A section requires financial data input. Use interactive mode (cli chat) to provide financial data.*
"""
        
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(output_text)
            console.print(f"[green]Output saved to {output}[/green]")
        else:
            console.print()
            console.print(Markdown(output_text))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def companies():
    """List available companies."""
    console.print("[bold]Available companies for 10-K generation:[/bold]\n")
    for ticker, info in TARGET_COMPANIES.items():
        console.print(f"  [cyan]{ticker}[/cyan] - {info['name']} (CIK: {info['cik']})")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the FastAPI server."""
    from src.api import start_server
    console.print(f"[bold]Starting API server on {host}:{port}...[/bold]")
    start_server(host=host, port=port)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()

