"""Planner/Orchestrator: Root Agent that plans and delegates to sub-agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import structlog

from aio.core.agent_instance import AgentInstance
from aio.core.message_bus import Message

logger = structlog.get_logger()


class AgentOrchestrator:
    """Root Agent: receives a user message, decides to handle directly or build a DAG."""

    def __init__(
        self,
        llm_registry: Any,
        mcp_client: Any,
        skill_engine: Any,
        context_engine: Any,
        autonomy_manager: Any,
    ) -> None:
        self.llm_registry = llm_registry
        self.mcp_client = mcp_client
        self.skill_engine = skill_engine
        self.context_engine = context_engine
        self.autonomy_manager = autonomy_manager

    async def run(self, message: Message, history: list[Message]) -> Message:
        """Process an incoming message.

        Simple tasks → handle directly via ReAct loop.
        Complex tasks → call build_dag tool → delegate to DAGRunner (Phase 2).
        """
        logger.info("orchestrator.run", session_id=message.session_id)
        # Phase 1: sequential processing via a single agent instance
        agent = AgentInstance(
            agent_id="root",
            template_name="general",
            llm_registry=self.llm_registry,
            mcp_client=self.mcp_client,
        )
        result = await agent.run(task=message.content, tools=[], history=history)
        return Message(
            role=message.role.__class__("assistant"),
            content=result,
            session_id=message.session_id,
            agent_name="root",
        )
