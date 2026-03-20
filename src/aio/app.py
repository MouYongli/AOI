"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

from aio import __version__
from aio.config.settings import load_config

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown."""
    config = load_config()
    logger.info("aio.startup", version=__version__)
    # TODO: initialize Database, LLMRegistry, MCPClient, MessageBus, Channels
    yield
    logger.info("aio.shutdown")


app = FastAPI(
    title="AIO — AI On-premise",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
