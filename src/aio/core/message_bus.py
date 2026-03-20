"""MessageBus: system hub routing messages from channels to agents and back."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    session_id: str
    channel_type: str = ""
    agent_name: str | None = None
    tool_call_id: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class MessageBus:
    """Central message bus: Channel → Root Agent → response pipeline."""

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}
        self._queue: asyncio.Queue[Message] = asyncio.Queue()

    def register_handler(self, channel_type: str, handler: Any) -> None:
        self._handlers[channel_type] = handler

    async def dispatch(self, message: Message) -> Message:
        """Dispatch an incoming user message to the orchestrator and return the response."""
        logger.info("message_bus.dispatch", session_id=message.session_id, role=message.role)
        # Placeholder: will be wired to AgentOrchestrator
        return Message(
            role=MessageRole.ASSISTANT,
            content="[MessageBus] Not yet connected to orchestrator.",
            session_id=message.session_id,
        )
