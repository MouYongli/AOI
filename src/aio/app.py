"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

from aio import __version__
from aio.config.settings import load_config
from aio.core.context_engine import ContextEngine
from aio.core.message_bus import MessageBus
from aio.core.orchestrator import AgentOrchestrator
from aio.core.skill_engine import SkillEngine
from aio.llm.registry import LLMRegistry
from aio.mcp.client import MCPClient

logger = structlog.get_logger()

_message_bus = MessageBus()
_telegram_adapter = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown."""
    global _telegram_adapter

    config = load_config()
    logger.info("aio.startup", version=__version__)

    # Initialize core components
    llm_registry = LLMRegistry.from_config(config.llm_providers, config.default_llm_provider)
    mcp_client = MCPClient()
    context_engine = ContextEngine()
    skill_engine = SkillEngine()

    # Build agent template lookup
    agent_templates = {t.name: t for t in config.agent_templates}

    # Connect MCP servers
    for server_cfg in config.mcp_servers:
        if server_cfg.enabled:
            await mcp_client.connect(server_cfg.name, server_cfg.command, server_cfg.args or None)

    # Wire orchestrator to message bus
    orchestrator = AgentOrchestrator(
        llm_registry=llm_registry,
        mcp_client=mcp_client,
        skill_engine=skill_engine,
        context_engine=context_engine,
        agent_templates=agent_templates,
    )
    _message_bus.set_orchestrator(orchestrator)

    # Start Telegram bot if enabled
    if config.telegram.enabled and config.telegram.bot_token:
        from aio.channels.telegram import TelegramAdapter

        _telegram_adapter = TelegramAdapter(
            bot_token=config.telegram.bot_token,
            allowed_user_ids=config.telegram.allowed_user_ids or None,
            message_bus=_message_bus,
        )
        await _telegram_adapter.start()
        logger.info("aio.telegram_started")

    yield

    # Shutdown
    if _telegram_adapter is not None:
        await _telegram_adapter.stop()
        logger.info("aio.telegram_stopped")

    logger.info("aio.shutdown")


app = FastAPI(
    title="AIO — AI On-premise",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
