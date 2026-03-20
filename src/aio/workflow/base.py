"""Base workflow runner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    id: str
    agent_template: str
    task: str
    dependencies: list[str] = field(default_factory=list)
    result: str | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING


@dataclass
class Workflow:
    id: str
    steps: list[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING


class WorkflowRunner(ABC):
    """Abstract workflow runner."""

    @abstractmethod
    async def run(self, workflow: Workflow, progress_cb: Any = None) -> dict[str, Any]:
        """Execute a workflow and return results."""

    @abstractmethod
    async def get_status(self, workflow_id: str) -> WorkflowStatus:
        """Get the current status of a workflow."""

    @abstractmethod
    async def cancel(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
