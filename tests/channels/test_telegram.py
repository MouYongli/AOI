"""Tests for the Telegram channel adapter.

Telegram's python-telegram-bot has heavy native dependencies (cryptography/cffi).
We mock the telegram package so tests can run in any environment.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Mock the telegram package before importing our module ────────────

_telegram_mock = types.ModuleType("telegram")
_telegram_mock.Bot = MagicMock  # type: ignore[attr-defined]
_telegram_mock.Update = MagicMock  # type: ignore[attr-defined]

_constants_mock = types.ModuleType("telegram.constants")
_constants_mock.ChatAction = MagicMock()  # type: ignore[attr-defined]
_constants_mock.ChatAction.TYPING = "typing"
_constants_mock.ParseMode = MagicMock()  # type: ignore[attr-defined]
_constants_mock.ParseMode.MARKDOWN = "Markdown"

_ext_mock = types.ModuleType("telegram.ext")
_ext_mock.Application = MagicMock  # type: ignore[attr-defined]
_ext_mock.CommandHandler = MagicMock  # type: ignore[attr-defined]
_ext_mock.ContextTypes = MagicMock  # type: ignore[attr-defined]
_ext_mock.MessageHandler = MagicMock  # type: ignore[attr-defined]
_ext_mock.filters = MagicMock  # type: ignore[attr-defined]

sys.modules.setdefault("telegram", _telegram_mock)
sys.modules.setdefault("telegram.constants", _constants_mock)
sys.modules.setdefault("telegram.ext", _ext_mock)

from aio.channels.telegram import (  # noqa: E402
    TelegramAdapter,
    TelegramProgressReporter,
    _get_or_create_session,
    _pending_confirmations,
    _user_session_list,
    _user_sessions,
)
from aio.core.message_bus import Message, MessageRole  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_session_state():
    """Clear global session state between tests."""
    _user_sessions.clear()
    _user_session_list.clear()
    _pending_confirmations.clear()
    yield
    _user_sessions.clear()
    _user_session_list.clear()
    _pending_confirmations.clear()


# ── Session management ───────────────────────────────────────────────


class TestSessionManagement:
    def test_get_or_create_session_creates_new(self):
        session_id = _get_or_create_session(12345)
        assert session_id
        assert 12345 in _user_sessions
        assert session_id in _user_session_list[12345]

    def test_get_or_create_session_returns_existing(self):
        s1 = _get_or_create_session(12345)
        s2 = _get_or_create_session(12345)
        assert s1 == s2

    def test_different_users_get_different_sessions(self):
        s1 = _get_or_create_session(111)
        s2 = _get_or_create_session(222)
        assert s1 != s2


# ── Access control ───────────────────────────────────────────────────


class TestAccessControl:
    def test_empty_allowlist_allows_all(self):
        adapter = TelegramAdapter(bot_token="test", allowed_user_ids=None)
        assert adapter._is_allowed(99999) is True

    def test_allowlist_blocks_unauthorized(self):
        adapter = TelegramAdapter(bot_token="test", allowed_user_ids=[111, 222])
        assert adapter._is_allowed(111) is True
        assert adapter._is_allowed(333) is False


# ── Progress reporter ────────────────────────────────────────────────


class TestTelegramProgressReporter:
    @pytest.mark.asyncio
    async def test_start_edits_message(self):
        bot = AsyncMock()
        reporter = TelegramProgressReporter(bot=bot, chat_id=1, message_id=10)
        await reporter.start("Loading…")
        bot.edit_message_text.assert_called_once()
        call_kwargs = bot.edit_message_text.call_args.kwargs
        assert "⏳ Loading…" in call_kwargs["text"]

    @pytest.mark.asyncio
    async def test_update_appends_line(self):
        bot = AsyncMock()
        reporter = TelegramProgressReporter(bot=bot, chat_id=1, message_id=10)
        await reporter.start("Start")
        await reporter.update("Step 1")
        assert bot.edit_message_text.call_count == 2
        last_text = bot.edit_message_text.call_args.kwargs["text"]
        assert "Step 1" in last_text

    @pytest.mark.asyncio
    async def test_finish_adds_done(self):
        bot = AsyncMock()
        reporter = TelegramProgressReporter(bot=bot, chat_id=1, message_id=10)
        await reporter.start("Start")
        await reporter.finish()
        last_text = bot.edit_message_text.call_args.kwargs["text"]
        assert "✅ Done" in last_text

    @pytest.mark.asyncio
    async def test_edit_failure_does_not_raise(self):
        bot = AsyncMock()
        bot.edit_message_text.side_effect = Exception("network error")
        reporter = TelegramProgressReporter(bot=bot, chat_id=1, message_id=10)
        await reporter.start("Test")  # Should not raise


# ── Adapter: send_response / send_confirmation ──────────────────────


class TestTelegramAdapterResponses:
    @pytest.mark.asyncio
    async def test_send_response_no_chat_mapping(self):
        adapter = TelegramAdapter(bot_token="test")
        msg = Message(role=MessageRole.ASSISTANT, content="hi", session_id="unknown")
        await adapter.send_response(msg)

    @pytest.mark.asyncio
    async def test_send_response_with_chat_mapping(self):
        adapter = TelegramAdapter(bot_token="test")
        mock_app = MagicMock()
        mock_app.bot = AsyncMock()
        adapter._app = mock_app
        adapter._session_chat_map["sess1"] = 42

        msg = Message(role=MessageRole.ASSISTANT, content="hello", session_id="sess1")
        await adapter.send_response(msg)
        mock_app.bot.send_message.assert_called_once()
        call_kwargs = mock_app.bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == 42
        assert call_kwargs["text"] == "hello"

    @pytest.mark.asyncio
    async def test_send_confirmation_timeout(self):
        adapter = TelegramAdapter(bot_token="test")
        mock_app = MagicMock()
        mock_app.bot = AsyncMock()
        adapter._app = mock_app
        adapter._session_chat_map["sess1"] = 42

        with patch("aio.channels.telegram.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await adapter.send_confirmation("Do this?", "sess1")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_typing_indicator(self):
        adapter = TelegramAdapter(bot_token="test")
        mock_app = MagicMock()
        mock_app.bot = AsyncMock()
        adapter._app = mock_app
        adapter._session_chat_map["sess1"] = 42

        await adapter.send_typing_indicator("sess1")
        mock_app.bot.send_chat_action.assert_called_once()
