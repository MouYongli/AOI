"""CLI channel adapter: aio chat / aio ask / pipe input."""

from __future__ import annotations

import structlog
from rich.console import Console
from rich.status import Status

from aio.channels.base import ChannelAdapter
from aio.channels.progress_reporter import ProgressReporter
from aio.core.message_bus import Message

logger = structlog.get_logger()

console = Console()


class CLIProgressReporter(ProgressReporter):
    """Progress reporter using Rich Status spinner."""

    def __init__(self) -> None:
        self._status: Status | None = None

    async def start(self, hint: str) -> None:
        self._status = console.status(hint)
        self._status.start()

    async def update(self, line: str) -> None:
        if self._status:
            self._status.update(line)

    async def finish(self) -> None:
        if self._status:
            self._status.stop()
            self._status = None


class CLIAdapter(ChannelAdapter):
    """CLI channel adapter.

    Supports:
    - `aio chat` — interactive REPL
    - `aio ask "question"` — single-shot
    - Pipe input: `echo "question" | aio ask`
    - `--agent` parameter to select agent template
    """

    async def start(self) -> None:
        logger.info("cli.start")

    async def stop(self) -> None:
        logger.info("cli.stop")

    async def send_response(self, message: Message) -> None:
        console.print(message.content)

    async def send_confirmation(self, prompt: str, session_id: str) -> bool:
        response = console.input(f"{prompt} [y/N]: ")
        return response.strip().lower() in ("y", "yes")

    async def send_typing_indicator(self, session_id: str) -> None:
        pass
