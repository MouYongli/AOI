"""Base channel adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aio.core.message_bus import Message


class ChannelAdapter(ABC):
    """Abstract base for all channel adapters (Telegram, CLI, Web, etc.)."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening for incoming messages."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""

    @abstractmethod
    async def send_response(self, message: Message) -> None:
        """Send a response message back to the user."""

    @abstractmethod
    async def send_confirmation(self, prompt: str, session_id: str) -> bool:
        """Request user confirmation for a tool call. Returns True if confirmed."""

    @abstractmethod
    async def send_typing_indicator(self, session_id: str) -> None:
        """Show a typing/processing indicator to the user."""
