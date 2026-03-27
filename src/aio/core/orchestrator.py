"""Planner/Orchestrator: Root Agent that plans and delegates to sub-agents."""

from __future__ import annotations

import uuid
from typing import Any

import structlog

from aio.config.settings import AgentTemplateConfig
from aio.core.agent_instance import AgentInstance
from aio.core.context_engine import ContextEngine
from aio.core.message_bus import Message, MessageRole
from aio.core.skill_engine import SkillEngine
from aio.llm.base import LLMMessage
from aio.llm.registry import LLMRegistry
from aio.mcp.client import MCPClient

logger = structlog.get_logger()

DEFAULT_SYSTEM_PROMPT = (
    "You are AIO, an AI assistant running on the user's own device. "
    "You are helpful, accurate, and concise. "
    "When you need to perform actions, use the available tools. "
    "Always explain what you are doing and present results clearly."
)


def _messages_to_llm_history(messages: list[Message]) -> list[LLMMessage]:
    """Convert MessageBus Messages to LLMMessage format for the provider."""
    return [
        LLMMessage(role=msg.role.value, content=msg.content)
        for msg in messages
    ]


class AgentOrchestrator:
    """Root Agent: receives a user message, decides to handle directly or build a DAG."""

    def __init__(
        self,
        llm_registry: LLMRegistry,
        mcp_client: MCPClient,
        skill_engine: SkillEngine,
        context_engine: ContextEngine,
        agent_templates: dict[str, AgentTemplateConfig] | None = None,
    ) -> None:
        self.llm_registry = llm_registry
        self.mcp_client = mcp_client
        self.skill_engine = skill_engine
        self.context_engine = context_engine
        self.agent_templates = agent_templates or {}

    def _resolve_template(self, template_name: str) -> AgentTemplateConfig:
        """Get agent template config, falling back to a default."""
        if template_name in self.agent_templates:
            return self.agent_templates[template_name]
        return AgentTemplateConfig(
            name="general",
            description="General assistant",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
        )

    async def run(
        self,
        message: Message,
        template_name: str = "general",
    ) -> Message:
        """Process an incoming message.

        Simple tasks → handle directly via ReAct loop.
        Complex tasks → call build_dag tool → delegate to DAGRunner (Phase 2).
        """
        logger.info("orchestrator.run", session_id=message.session_id)

        # Retrieve conversation history
        history_msgs = await self.context_engine.get_context(message.session_id)
        llm_history = _messages_to_llm_history(history_msgs)

        # Store user message in context
        await self.context_engine.add_message(message.session_id, message)

        # Resolve agent template
        template = self._resolve_template(template_name)
        system_prompt = template.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Resolve LLM provider
        provider = self.llm_registry.get(template.llm_provider)

        # Gather available tools from MCP
        tools = await self.mcp_client.list_tools()

        # Create and run agent instance
        agent = AgentInstance(
            agent_id=f"root-{uuid.uuid4().hex[:8]}",
            template_name=template.name,
            llm_provider=provider,
            mcp_client=self.mcp_client,
            system_prompt=system_prompt,
            max_loops=template.max_react_loops,
        )
        result = await agent.run(
            task=message.content,
            tools=tools,
            history=llm_history,
        )

        # Build response message
        response = Message(
            role=MessageRole.ASSISTANT,
            content=result,
            session_id=message.session_id,
            agent_name=template.name,
        )

        # Store assistant response in context
        await self.context_engine.add_message(message.session_id, response)

        return response
