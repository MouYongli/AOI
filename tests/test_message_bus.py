"""Tests for MessageBus."""

import pytest

from aio.core.message_bus import Message, MessageBus, MessageRole


@pytest.mark.asyncio
async def test_dispatch_returns_response() -> None:
    bus = MessageBus()
    msg = Message(role=MessageRole.USER, content="hello", session_id="s1")
    response = await bus.dispatch(msg)
    assert response.role == MessageRole.ASSISTANT
    assert response.session_id == "s1"
