"""Tests for ContextEngine."""

import pytest

from aio.core.context_engine import ContextEngine
from aio.core.message_bus import Message, MessageRole


@pytest.mark.asyncio
async def test_add_and_get_context() -> None:
    engine = ContextEngine()
    msg = Message(role=MessageRole.USER, content="hi", session_id="s1")
    await engine.add_message("s1", msg)

    history = await engine.get_context("s1")
    assert len(history) == 1
    assert history[0].content == "hi"


@pytest.mark.asyncio
async def test_empty_context() -> None:
    engine = ContextEngine()
    history = await engine.get_context("nonexistent")
    assert history == []
