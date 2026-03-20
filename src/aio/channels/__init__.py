"""Channel adapters: Telegram, CLI, progress reporter."""

from aio.channels.base import ChannelAdapter
from aio.channels.cli_adapter import CLIAdapter, CLIProgressReporter
from aio.channels.progress_reporter import NoOpProgressReporter, ProgressReporter

__all__ = [
    "ChannelAdapter",
    "CLIAdapter",
    "CLIProgressReporter",
    "NoOpProgressReporter",
    "ProgressReporter",
    "TelegramAdapter",
    "TelegramProgressReporter",
]


def __getattr__(name: str):  # noqa: N807
    """Lazy-import Telegram classes to avoid hard dependency on python-telegram-bot."""
    if name in ("TelegramAdapter", "TelegramProgressReporter"):
        from aio.channels.telegram import TelegramAdapter, TelegramProgressReporter

        return TelegramAdapter if name == "TelegramAdapter" else TelegramProgressReporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
