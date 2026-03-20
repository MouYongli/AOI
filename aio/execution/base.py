"""Base execution environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    is_error: bool = False


class ExecutionEnvironment(ABC):
    """Abstract execution environment for tool calls."""

    @abstractmethod
    async def execute(self, command: str, **kwargs: Any) -> ExecutionResult:
        """Execute a command in this environment."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the execution environment is available."""
