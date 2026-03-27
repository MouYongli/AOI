"""Tests for AgentInstance ReAct loop and Orchestrator integration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aio.core.agent_instance import AgentInstance, AgentStatus
from aio.core.context_engine import ContextEngine
from aio.core.message_bus import Message, MessageBus, MessageRole
from aio.core.orchestrator import AgentOrchestrator
from aio.core.skill_engine import SkillEngine
from aio.llm.base import LLMMessage, LLMProvider, LLMResponse
from aio.mcp.client import MCPClient, MCPTool, MCPToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeLLMProvider(LLMProvider):
    """LLM provider that returns scripted responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp

    async def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeLLMRegistry:
    """Minimal registry returning a single provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def get(self, name: str | None = None) -> LLMProvider:
        return self._provider


# ---------------------------------------------------------------------------
# AgentInstance tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_react_no_tool_calls() -> None:
    """LLM returns text only → single step, no tool loop."""
    provider = FakeLLMProvider([
        LLMResponse(content="Hello! How can I help?", tool_calls=[]),
    ])
    agent = AgentInstance(
        agent_id="test-1",
        template_name="general",
        llm_provider=provider,
    )
    result = await agent.run(task="Hi")
    assert result == "Hello! How can I help?"
    assert agent.status == AgentStatus.DONE


@pytest.mark.asyncio
async def test_react_with_tool_call() -> None:
    """LLM requests a tool → MCP executes → LLM gives final answer."""
    provider = FakeLLMProvider([
        # Step 1: LLM requests a tool call
        LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "name": "websearch", "arguments": {"query": "weather"}}],
        ),
        # Step 2: LLM gives final answer after observation
        LLMResponse(content="It's sunny today!", tool_calls=[]),
    ])

    mcp = MCPClient()
    mcp.call_tool = AsyncMock(return_value=MCPToolResult(content="Sunny, 25°C"))

    tools = [MCPTool(name="websearch", description="Search the web", input_schema={})]

    agent = AgentInstance(
        agent_id="test-2",
        template_name="general",
        llm_provider=provider,
        mcp_client=mcp,
    )
    result = await agent.run(task="What's the weather?", tools=tools)
    assert result == "It's sunny today!"
    assert agent.status == AgentStatus.DONE
    mcp.call_tool.assert_called_once_with("websearch", {"query": "weather"})


@pytest.mark.asyncio
async def test_react_multi_step_tool_calls() -> None:
    """LLM chains two tool calls before final answer."""
    provider = FakeLLMProvider([
        LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "name": "search", "arguments": {"q": "info"}}],
        ),
        LLMResponse(
            content="",
            tool_calls=[{"id": "tc2", "name": "read", "arguments": {"path": "/tmp/f"}}],
        ),
        LLMResponse(content="Here is the result.", tool_calls=[]),
    ])

    mcp = MCPClient()
    mcp.call_tool = AsyncMock(side_effect=[
        MCPToolResult(content="found doc"),
        MCPToolResult(content="file contents"),
    ])

    agent = AgentInstance(
        agent_id="test-3",
        template_name="general",
        llm_provider=provider,
        mcp_client=mcp,
    )
    result = await agent.run(task="Research and read")
    assert result == "Here is the result."
    assert mcp.call_tool.call_count == 2


@pytest.mark.asyncio
async def test_react_max_loops_exceeded() -> None:
    """When max loops exceeded, agent asks LLM for summary."""
    # Always return tool calls
    always_tool = LLMResponse(
        content="",
        tool_calls=[{"id": "tc", "name": "loop_tool", "arguments": {}}],
    )
    final = LLMResponse(content="Summary after max loops.", tool_calls=[])

    provider = FakeLLMProvider([always_tool, always_tool, final])

    mcp = MCPClient()
    mcp.call_tool = AsyncMock(return_value=MCPToolResult(content="ok"))

    agent = AgentInstance(
        agent_id="test-4",
        template_name="general",
        llm_provider=provider,
        mcp_client=mcp,
        max_loops=2,
    )
    result = await agent.run(task="Loop forever")
    assert result == "Summary after max loops."


@pytest.mark.asyncio
async def test_react_tool_error() -> None:
    """Tool returns error → error is passed to LLM as observation."""
    provider = FakeLLMProvider([
        LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "name": "fail_tool", "arguments": {}}],
        ),
        LLMResponse(content="The tool failed, but I can help anyway.", tool_calls=[]),
    ])

    mcp = MCPClient()
    mcp.call_tool = AsyncMock(return_value=MCPToolResult(content="Permission denied", is_error=True))

    agent = AgentInstance(
        agent_id="test-5",
        template_name="general",
        llm_provider=provider,
        mcp_client=mcp,
    )
    result = await agent.run(task="Try something")
    assert "tool failed" in result.lower()


@pytest.mark.asyncio
async def test_react_no_mcp_client() -> None:
    """Without MCP client, tool calls produce 'not available' observation."""
    provider = FakeLLMProvider([
        LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "name": "some_tool", "arguments": {}}],
        ),
        LLMResponse(content="No tools available, answering directly.", tool_calls=[]),
    ])

    agent = AgentInstance(
        agent_id="test-6",
        template_name="general",
        llm_provider=provider,
        mcp_client=None,
    )
    result = await agent.run(task="Use a tool")
    assert result == "No tools available, answering directly."


@pytest.mark.asyncio
async def test_react_llm_error_sets_status() -> None:
    """LLM provider raising an error → agent status becomes ERROR."""
    provider = FakeLLMProvider([])
    provider.chat = AsyncMock(side_effect=RuntimeError("LLM is down"))

    agent = AgentInstance(
        agent_id="test-7",
        template_name="general",
        llm_provider=provider,
    )
    with pytest.raises(RuntimeError, match="LLM is down"):
        await agent.run(task="Hello")
    assert agent.status == AgentStatus.ERROR


@pytest.mark.asyncio
async def test_system_prompt_included() -> None:
    """System prompt is prepended to messages sent to LLM."""
    calls: list[list[LLMMessage]] = []

    async def capture_chat(messages, tools=None, stream=False):
        calls.append(messages)
        return LLMResponse(content="ok")

    provider = FakeLLMProvider([])
    provider.chat = capture_chat

    agent = AgentInstance(
        agent_id="test-8",
        template_name="general",
        llm_provider=provider,
        system_prompt="You are a helpful bot.",
    )
    await agent.run(task="Hi")
    assert calls[0][0].role == "system"
    assert calls[0][0].content == "You are a helpful bot."


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_end_to_end() -> None:
    """Orchestrator wires everything: context, LLM, and returns response."""
    provider = FakeLLMProvider([
        LLMResponse(content="I can help with that!", tool_calls=[]),
    ])
    registry = FakeLLMRegistry(provider)
    mcp = MCPClient()
    context = ContextEngine()
    skill = SkillEngine()

    orch = AgentOrchestrator(
        llm_registry=registry,
        mcp_client=mcp,
        skill_engine=skill,
        context_engine=context,
    )

    msg = Message(role=MessageRole.USER, content="Help me", session_id="s1")
    response = await orch.run(msg)

    assert response.role == MessageRole.ASSISTANT
    assert response.content == "I can help with that!"
    assert response.session_id == "s1"

    # Verify context was stored
    history = await context.get_context("s1")
    assert len(history) == 2  # user msg + assistant response


@pytest.mark.asyncio
async def test_orchestrator_preserves_history() -> None:
    """Multiple messages in same session build conversation history."""
    provider = FakeLLMProvider([
        LLMResponse(content="Response 1"),
        LLMResponse(content="Response 2"),
    ])
    registry = FakeLLMRegistry(provider)
    orch = AgentOrchestrator(
        llm_registry=registry,
        mcp_client=MCPClient(),
        skill_engine=SkillEngine(),
        context_engine=ContextEngine(),
    )

    msg1 = Message(role=MessageRole.USER, content="First", session_id="s1")
    await orch.run(msg1)

    msg2 = Message(role=MessageRole.USER, content="Second", session_id="s1")
    resp2 = await orch.run(msg2)

    assert resp2.content == "Response 2"


# ---------------------------------------------------------------------------
# MessageBus integration tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_message_bus_with_orchestrator() -> None:
    """MessageBus dispatches to orchestrator when wired."""
    provider = FakeLLMProvider([
        LLMResponse(content="Real response from orchestrator"),
    ])
    registry = FakeLLMRegistry(provider)
    orch = AgentOrchestrator(
        llm_registry=registry,
        mcp_client=MCPClient(),
        skill_engine=SkillEngine(),
        context_engine=ContextEngine(),
    )

    bus = MessageBus()
    bus.set_orchestrator(orch)

    msg = Message(role=MessageRole.USER, content="Hello", session_id="s1")
    response = await bus.dispatch(msg)

    assert response.content == "Real response from orchestrator"
    assert response.role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_message_bus_fallback_echo() -> None:
    """MessageBus without orchestrator returns echo fallback."""
    bus = MessageBus()
    msg = Message(role=MessageRole.USER, content="hello", session_id="s1")
    response = await bus.dispatch(msg)
    assert response.role == MessageRole.ASSISTANT
    assert "hello" in response.content
    assert response.session_id == "s1"
