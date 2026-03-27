"""AgentInstance: runtime agent with ReAct loop."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse
from aio.mcp.client import MCPClient, MCPTool

logger = structlog.get_logger()


class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


def _mcp_tools_to_llm_schema(tools: list[MCPTool]) -> list[dict[str, Any]]:
    """Convert MCPTool list to the JSON-schema tool format expected by LLM providers."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def _build_messages(
    system_prompt: str,
    history: list[LLMMessage],
    task: str,
) -> list[LLMMessage]:
    """Build the message list for the LLM call."""
    messages: list[LLMMessage] = []
    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))
    messages.extend(history)
    messages.append(LLMMessage(role="user", content=task))
    return messages


class AgentInstance:
    """A runtime Agent instance (spawned from template or dynamically generated).

    Runs an independent ReAct loop: Reason → Act (tool call) → Observe → repeat.
    """

    MAX_REACT_LOOPS = 10

    def __init__(
        self,
        agent_id: str,
        template_name: str,
        llm_provider: LLMProvider,
        mcp_client: MCPClient | None = None,
        parent_id: str | None = None,
        system_prompt: str = "",
        max_loops: int | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.template_name = template_name
        self.llm_provider = llm_provider
        self.mcp_client = mcp_client
        self.parent_id = parent_id
        self.system_prompt = system_prompt
        self.status = AgentStatus.PENDING
        if max_loops is not None:
            self.MAX_REACT_LOOPS = max_loops

    async def run(
        self,
        task: str,
        tools: list[MCPTool] | None = None,
        history: list[LLMMessage] | None = None,
    ) -> str:
        """Execute ReAct loop for the given task.

        1. Call LLM with conversation history + available tools
        2. If LLM returns tool_calls → execute via MCP → feed observations back
        3. Repeat until LLM returns a final text response or max loops reached
        """
        self.status = AgentStatus.RUNNING
        logger.info("agent.run", agent_id=self.agent_id, template=self.template_name)

        tools = tools or []
        history = list(history or [])
        tool_schemas = _mcp_tools_to_llm_schema(tools) if tools else None

        # Build initial messages
        messages = _build_messages(self.system_prompt, history, task)

        try:
            for step in range(self.MAX_REACT_LOOPS):
                logger.debug("agent.react_step", agent_id=self.agent_id, step=step)

                response: LLMResponse = await self.llm_provider.chat(
                    messages=messages,
                    tools=tool_schemas,
                )

                # No tool calls → final answer
                if not response.tool_calls:
                    self.status = AgentStatus.DONE
                    logger.info(
                        "agent.done",
                        agent_id=self.agent_id,
                        steps=step + 1,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    )
                    return response.content

                # Append assistant message with tool calls
                messages.append(LLMMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ))

                # Execute each tool call and collect observations
                for tc in response.tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})
                    tool_call_id = tc.get("id", "")

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    logger.info("agent.tool_call", tool=tool_name, args=tool_args)

                    if self.mcp_client:
                        result = await self.mcp_client.call_tool(tool_name, tool_args)
                        observation = result.content
                        if result.is_error:
                            observation = f"[ERROR] {observation}"
                    else:
                        observation = f"[Tool '{tool_name}' not available — no MCP client configured]"

                    messages.append(LLMMessage(
                        role="tool",
                        content=observation,
                        tool_call_id=tool_call_id,
                    ))

            # Exceeded max loops — ask LLM for a final summary
            messages.append(LLMMessage(
                role="user",
                content="You have reached the maximum number of tool call iterations. Please provide your final answer based on the information gathered so far.",
            ))
            response = await self.llm_provider.chat(messages=messages, tools=None)
            self.status = AgentStatus.DONE
            return response.content

        except Exception as exc:
            self.status = AgentStatus.ERROR
            logger.error("agent.error", agent_id=self.agent_id, error=str(exc))
            raise
