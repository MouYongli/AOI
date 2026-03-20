"""Telegram Bot adapter: multi-session, real-time progress via edit_text."""

from __future__ import annotations

from typing import Any

import structlog

from aio.channels.base import ChannelAdapter
from aio.channels.progress_reporter import ProgressReporter
from aio.core.message_bus import Message

logger = structlog.get_logger()


class TelegramProgressReporter(ProgressReporter):
    """Progress reporter that edits a Telegram placeholder message in-place."""

    def __init__(self, bot: Any, chat_id: int, message_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id
        self._lines: list[str] = []

    async def start(self, hint: str) -> None:
        self._lines = [f"⏳ {hint}"]
        # TODO: edit_text the placeholder message

    async def update(self, line: str) -> None:
        self._lines.append(line)
        # TODO: edit_text with updated content

    async def finish(self) -> None:
        self._lines.append("✅ Done")
        # TODO: edit_text with final content


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot channel adapter.

    Features:
    - Multi-session: /new, /switch, /sessions commands
    - Text, image, file, and voice input
    - Placeholder message with real-time progress (edit_text)
    """

    def __init__(self, bot_token: str, allowed_user_ids: list[int] | None = None) -> None:
        self._bot_token = bot_token
        self._allowed_user_ids = allowed_user_ids or []

    async def start(self) -> None:
        logger.info("telegram.start")
        # TODO: initialize python-telegram-bot Application and register handlers

    async def stop(self) -> None:
        logger.info("telegram.stop")

    async def send_response(self, message: Message) -> None:
        logger.info("telegram.send_response", session_id=message.session_id)

    async def send_confirmation(self, prompt: str, session_id: str) -> bool:
        logger.info("telegram.send_confirmation", session_id=session_id)
        return False

    async def send_typing_indicator(self, session_id: str) -> None:
        pass
