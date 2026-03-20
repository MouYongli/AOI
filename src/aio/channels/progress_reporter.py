"""ProgressReporter: abstract interface for real-time progress updates."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ProgressReporter(ABC):
    """Push progress updates to the user during long-running operations."""

    @abstractmethod
    async def start(self, hint: str) -> None:
        """Signal that a new operation has started."""

    @abstractmethod
    async def update(self, line: str) -> None:
        """Push an incremental progress line."""

    @abstractmethod
    async def finish(self) -> None:
        """Signal that the operation is complete."""


class NoOpProgressReporter(ProgressReporter):
    """Silent reporter for non-interactive contexts."""

    async def start(self, hint: str) -> None:
        pass

    async def update(self, line: str) -> None:
        pass

    async def finish(self) -> None:
        pass
