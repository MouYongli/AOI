"""Telegram Bot adapter: multi-session, real-time progress via edit_text."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import structlog
from telegram import Bot, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from aio.channels.base import ChannelAdapter
from aio.channels.progress_reporter import ProgressReporter
from aio.core.message_bus import Message, MessageBus, MessageRole

logger = structlog.get_logger()

# Per-user session mapping: user_id -> active session_id
_user_sessions: dict[int, str] = {}
# All sessions per user: user_id -> list of session_ids
_user_session_list: dict[int, list[str]] = {}
# Pending confirmations: session_id -> asyncio.Future[bool]
_pending_confirmations: dict[str, asyncio.Future[bool]] = {}


def _get_or_create_session(user_id: int) -> str:
    """Return the active session for a user, creating one if needed."""
    if user_id not in _user_sessions:
        session_id = str(uuid.uuid4())
        _user_sessions[user_id] = session_id
        _user_session_list.setdefault(user_id, []).append(session_id)
    return _user_sessions[user_id]


class TelegramProgressReporter(ProgressReporter):
    """Progress reporter that edits a Telegram placeholder message in-place."""

    def __init__(self, bot: Bot, chat_id: int, message_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._message_id = message_id
        self._lines: list[str] = []

    async def start(self, hint: str) -> None:
        self._lines = [f"⏳ {hint}"]
        await self._edit()

    async def update(self, line: str) -> None:
        self._lines.append(line)
        await self._edit()

    async def finish(self) -> None:
        self._lines.append("✅ Done")
        await self._edit()

    async def _edit(self) -> None:
        text = "\n".join(self._lines)
        try:
            await self._bot.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=text,
            )
        except Exception:
            logger.debug("telegram.progress_edit_failed", chat_id=self._chat_id)


class TelegramAdapter(ChannelAdapter):
    """Telegram Bot channel adapter.

    Features:
    - Multi-session: /new, /switch, /sessions commands
    - Access control via allowed_user_ids
    - Placeholder message with real-time progress (edit_text)
    - Inline confirmation for tool calls (callback query)
    """

    def __init__(
        self,
        bot_token: str,
        allowed_user_ids: list[int] | None = None,
        message_bus: MessageBus | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._allowed_user_ids = set(allowed_user_ids) if allowed_user_ids else set()
        self._message_bus = message_bus
        self._app: Application | None = None  # type: ignore[type-arg]
        # chat_id -> session_id for response routing
        self._session_chat_map: dict[str, int] = {}

    # ── Access control ───────────────────────────────────────────────

    def _is_allowed(self, user_id: int) -> bool:
        """Check if the user is allowed. Empty set means allow all."""
        if not self._allowed_user_ids:
            return True
        return user_id in self._allowed_user_ids

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        logger.info("telegram.start")
        self._app = (
            Application.builder()
            .token(self._bot_token)
            .build()
        )

        # Register command handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("new", self._cmd_new_session))
        self._app.add_handler(CommandHandler("sessions", self._cmd_list_sessions))
        self._app.add_handler(CommandHandler("switch", self._cmd_switch_session))
        self._app.add_handler(CommandHandler("yes", self._cmd_confirm_yes))
        self._app.add_handler(CommandHandler("no", self._cmd_confirm_no))

        # Text message handler (must be last)
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text_message)
        )

        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram.polling_started")

    async def stop(self) -> None:
        logger.info("telegram.stop")
        if self._app:
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send_response(self, message: Message) -> None:
        chat_id = self._session_chat_map.get(message.session_id)
        if chat_id is None:
            logger.warning("telegram.no_chat_for_session", session_id=message.session_id)
            return
        if self._app is None:
            return
        await self._app.bot.send_message(
            chat_id=chat_id,
            text=message.content or "(empty response)",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def send_confirmation(self, prompt: str, session_id: str) -> bool:
        chat_id = self._session_chat_map.get(session_id)
        if chat_id is None or self._app is None:
            return False
        await self._app.bot.send_message(
            chat_id=chat_id,
            text=f"🔐 *Confirmation required*\n\n{prompt}\n\nReply /yes or /no",
            parse_mode=ParseMode.MARKDOWN,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        _pending_confirmations[session_id] = future
        try:
            return await asyncio.wait_for(future, timeout=120.0)
        except asyncio.TimeoutError:
            logger.info("telegram.confirmation_timeout", session_id=session_id)
            await self._app.bot.send_message(chat_id=chat_id, text="⏰ Confirmation timed out. Denied.")
            return False
        finally:
            _pending_confirmations.pop(session_id, None)

    async def send_typing_indicator(self, session_id: str) -> None:
        chat_id = self._session_chat_map.get(session_id)
        if chat_id is None or self._app is None:
            return
        await self._app.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # ── Command handlers ─────────────────────────────────────────────

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return
        if not self._is_allowed(update.effective_user.id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return
        session_id = _get_or_create_session(update.effective_user.id)
        self._session_chat_map[session_id] = update.message.chat_id
        await update.message.reply_text(
            "👋 Welcome to AIO!\n\n"
            "Send me a message to start chatting.\n\n"
            "Commands:\n"
            "/new — Start a new session\n"
            "/sessions — List your sessions\n"
            "/switch <n> — Switch to session #n\n"
            "/help — Show this help"
        )

    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await update.message.reply_text(
            "🤖 *AIO Bot Commands*\n\n"
            "/new — Start a new session\n"
            "/sessions — List your sessions\n"
            "/switch <n> — Switch to session #n\n"
            "/yes — Confirm a pending action\n"
            "/no — Deny a pending action\n"
            "/help — Show this help",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_new_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return
        if not self._is_allowed(update.effective_user.id):
            return
        user_id = update.effective_user.id
        session_id = str(uuid.uuid4())
        _user_sessions[user_id] = session_id
        _user_session_list.setdefault(user_id, []).append(session_id)
        self._session_chat_map[session_id] = update.message.chat_id
        idx = len(_user_session_list[user_id])
        await update.message.reply_text(f"✨ New session #{idx} created. You can start chatting.")

    async def _cmd_list_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return
        user_id = update.effective_user.id
        sessions = _user_session_list.get(user_id, [])
        active = _user_sessions.get(user_id)
        if not sessions:
            await update.message.reply_text("No sessions yet. Send a message to start one.")
            return
        lines = []
        for i, sid in enumerate(sessions, 1):
            marker = " 👈" if sid == active else ""
            lines.append(f"#{i} `{sid[:8]}…`{marker}")
        await update.message.reply_text(
            "📋 *Your sessions:*\n\n" + "\n".join(lines),
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_switch_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message:
            return
        user_id = update.effective_user.id
        sessions = _user_session_list.get(user_id, [])
        args = context.args or []
        if not args or not args[0].isdigit():
            await update.message.reply_text("Usage: /switch <session number>")
            return
        idx = int(args[0]) - 1
        if idx < 0 or idx >= len(sessions):
            await update.message.reply_text(f"Invalid session number. You have {len(sessions)} sessions.")
            return
        _user_sessions[user_id] = sessions[idx]
        self._session_chat_map[sessions[idx]] = update.message.chat_id
        await update.message.reply_text(f"🔄 Switched to session #{idx + 1}.")

    async def _cmd_confirm_yes(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._resolve_confirmation(update, True)

    async def _cmd_confirm_no(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._resolve_confirmation(update, False)

    async def _resolve_confirmation(self, update: Update, value: bool) -> None:
        if not update.effective_user or not update.message:
            return
        user_id = update.effective_user.id
        session_id = _user_sessions.get(user_id)
        if session_id and session_id in _pending_confirmations:
            future = _pending_confirmations[session_id]
            if not future.done():
                future.set_result(value)
            label = "Confirmed ✅" if value else "Denied ❌"
            await update.message.reply_text(label)
        else:
            await update.message.reply_text("No pending confirmation.")

    # ── Message handler ──────────────────────────────────────────────

    async def _on_text_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_user or not update.message or not update.message.text:
            return
        user_id = update.effective_user.id
        if not self._is_allowed(user_id):
            await update.message.reply_text("⛔ You are not authorized to use this bot.")
            return

        session_id = _get_or_create_session(user_id)
        self._session_chat_map[session_id] = update.message.chat_id

        # Send typing indicator
        await update.message.chat.send_action(ChatAction.TYPING)

        # Send a placeholder message for progress updates
        placeholder = await update.message.reply_text("⏳ Thinking…")
        reporter = TelegramProgressReporter(
            bot=self._app.bot if self._app else context.bot,
            chat_id=update.message.chat_id,
            message_id=placeholder.message_id,
        )

        # Build the user message
        user_msg = Message(
            role=MessageRole.USER,
            content=update.message.text,
            session_id=session_id,
            channel_type="telegram",
            metadata={
                "telegram_user_id": user_id,
                "telegram_chat_id": update.message.chat_id,
                "telegram_message_id": update.message.message_id,
            },
        )

        # Dispatch through message bus
        if self._message_bus:
            await reporter.start("Processing your message…")
            response = await self._message_bus.dispatch(user_msg)
            await reporter.finish()

            # Replace the placeholder with the final response
            try:
                await context.bot.edit_message_text(
                    chat_id=update.message.chat_id,
                    message_id=placeholder.message_id,
                    text=response.content or "(empty response)",
                    parse_mode=ParseMode.MARKDOWN,
                )
            except Exception:
                # Fallback: send as a new message if edit fails
                await update.message.reply_text(response.content or "(empty response)")
        else:
            await context.bot.edit_message_text(
                chat_id=update.message.chat_id,
                message_id=placeholder.message_id,
                text="⚠️ Bot is not connected to the message bus yet.",
            )
