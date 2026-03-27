"""CLI entry point: aio chat / aio ask / aio serve."""

from __future__ import annotations

import asyncio
import uuid

import typer
from rich.console import Console

app = typer.Typer(name="aio", help="AIO — AI On-premise CLI")
console = Console()


def _build_cli_components(config_path: str = "config/config.yaml"):
    """Initialize core components for CLI usage (no FastAPI needed)."""
    import os

    os.environ.setdefault("AIO_CONFIG_PATH", config_path)

    from aio.config.settings import load_config
    from aio.core.context_engine import ContextEngine
    from aio.core.message_bus import MessageBus
    from aio.core.orchestrator import AgentOrchestrator
    from aio.core.skill_engine import SkillEngine
    from aio.llm.registry import LLMRegistry
    from aio.mcp.client import MCPClient

    config = load_config(config_path)
    llm_registry = LLMRegistry.from_config(config.llm_providers, config.default_llm_provider)
    mcp_client = MCPClient()
    context_engine = ContextEngine()
    skill_engine = SkillEngine()
    agent_templates = {t.name: t for t in config.agent_templates}

    orchestrator = AgentOrchestrator(
        llm_registry=llm_registry,
        mcp_client=mcp_client,
        skill_engine=skill_engine,
        context_engine=context_engine,
        agent_templates=agent_templates,
    )

    bus = MessageBus()
    bus.set_orchestrator(orchestrator)
    return bus, mcp_client, config


@app.command()
def serve(
    config: str = typer.Option("config/config.yaml", "--config", "-c", help="Config file path"),
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
) -> None:
    """Start the AIO gateway (FastAPI + Telegram bot)."""
    import os

    import uvicorn

    # Set config path via env so app.py can pick it up
    os.environ.setdefault("AIO_CONFIG_PATH", config)

    console.print(f"[bold]AIO Gateway[/bold] starting on {host}:{port}")
    console.print(f"Config: {config}")
    uvicorn.run("aio.app:app", host=host, port=port, log_level="info")


@app.command()
def chat(
    agent: str = typer.Option("general", "--agent", "-a", help="Agent template to use"),
    config: str = typer.Option("config/config.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Start an interactive chat session."""
    console.print(f"[bold]AIO Chat[/bold] (agent: {agent})")
    console.print("Type 'exit' or Ctrl+C to quit.\n")

    bus, mcp_client, cfg = _build_cli_components(config)
    session_id = str(uuid.uuid4())

    async def _connect_mcp_servers():
        for server_cfg in cfg.mcp_servers:
            if server_cfg.enabled:
                await mcp_client.connect(server_cfg.name, server_cfg.command, server_cfg.args or None)

    asyncio.get_event_loop().run_until_complete(_connect_mcp_servers()) if False else None

    from aio.core.message_bus import Message, MessageRole

    async def _dispatch(user_input: str) -> str:
        msg = Message(
            role=MessageRole.USER,
            content=user_input,
            session_id=session_id,
            channel_type="cli",
        )
        response = await bus.dispatch(msg)
        return response.content

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue
            with console.status("[bold blue]Thinking...[/bold blue]"):
                reply = asyncio.run(_dispatch(user_input))
            console.print(f"[bold blue]AIO:[/bold blue] {reply}\n")
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\nBye!")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    agent: str = typer.Option("general", "--agent", "-a", help="Agent template to use"),
    config: str = typer.Option("config/config.yaml", "--config", "-c", help="Config file path"),
) -> None:
    """Ask a single question and get a response."""
    bus, mcp_client, cfg = _build_cli_components(config)
    session_id = str(uuid.uuid4())

    from aio.core.message_bus import Message, MessageRole

    async def _dispatch() -> str:
        msg = Message(
            role=MessageRole.USER,
            content=question,
            session_id=session_id,
            channel_type="cli",
        )
        response = await bus.dispatch(msg)
        return response.content

    with console.status("[bold blue]Thinking...[/bold blue]"):
        reply = asyncio.run(_dispatch())
    console.print(f"[bold blue]AIO:[/bold blue] {reply}")


@app.command()
def version() -> None:
    """Show AIO version."""
    from aio import __version__

    console.print(f"AIO v{__version__}")


if __name__ == "__main__":
    app()
