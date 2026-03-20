"""AgentInstance: runtime agent with ReAct loop."""

from __future__ import annotations

from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class AgentInstance:
    """A runtime Agent instance (spawned from template or dynamically generated).

    Runs an independent ReAct loop: Reason → Act (tool call) → Observe → repeat.
    """

    MAX_REACT_LOOPS = 10

    def __init__(
        self,
        agent_id: str,
        template_name: str,
        llm_registry: Any,
        mcp_client: Any,
        parent_id: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.template_name = template_name
        self.llm_registry = llm_registry
        self.mcp_client = mcp_client
        self.parent_id = parent_id
        self.status = AgentStatus.PENDING

    async def run(self, task: str, tools: list[Any], history: list[Any]) -> str:
        """Execute ReAct loop for the given task."""
        self.status = AgentStatus.RUNNING
        logger.info("agent.run", agent_id=self.agent_id, template=self.template_name)

        try:
            # Placeholder ReAct loop
            for step in range(self.MAX_REACT_LOOPS):
                logger.debug("agent.react_step", agent_id=self.agent_id, step=step)
                # TODO: LLM call → tool call → observe
                break

            self.status = AgentStatus.DONE
            return f"[Agent {self.agent_id}] Task completed (placeholder)."
        except Exception as exc:
            self.status = AgentStatus.ERROR
            logger.error("agent.error", agent_id=self.agent_id, error=str(exc))
            raise
