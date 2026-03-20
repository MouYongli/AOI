"""CLI entry point: aio chat / aio ask / pipe input."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(name="aio", help="AIO — AI On-premise CLI")
console = Console()


@app.command()
def chat(
    agent: str = typer.Option("general", "--agent", "-a", help="Agent template to use"),
) -> None:
    """Start an interactive chat session."""
    console.print(f"[bold]AIO Chat[/bold] (agent: {agent})")
    console.print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            # TODO: wire to MessageBus
            console.print(f"[bold blue]AIO:[/bold blue] [placeholder response]")
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\nBye!")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    agent: str = typer.Option("general", "--agent", "-a", help="Agent template to use"),
) -> None:
    """Ask a single question and get a response."""
    console.print(f"[bold blue]AIO:[/bold blue] [placeholder response to: {question}]")


@app.command()
def version() -> None:
    """Show AIO version."""
    from aio import __version__

    console.print(f"AIO v{__version__}")


if __name__ == "__main__":
    app()
