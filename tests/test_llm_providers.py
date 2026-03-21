"""Tests for LLM provider implementations."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aio.llm.base import LLMMessage, LLMResponse, ToolCall
from aio.llm.claude_provider import ClaudeProvider, _extract_system, _to_anthropic_tools
from aio.llm.ollama import OllamaProvider, _to_ollama_tools
from aio.llm.openai_provider import OpenAIProvider, _to_openai_tools
from aio.llm.azure_openai import AzureOpenAIProvider
from aio.llm.registry import LLMRegistry
from aio.config.settings import LLMProviderConfig, LLMProviderType


# ---------------------------------------------------------------------------
# ToolCall dataclass
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_to_dict(self) -> None:
        tc = ToolCall(id="tc1", name="get_weather", arguments={"city": "Berlin"})
        d = tc.to_dict()
        assert d == {"id": "tc1", "name": "get_weather", "arguments": {"city": "Berlin"}}

    def test_from_dict(self) -> None:
        tc = ToolCall.from_dict({"id": "tc2", "name": "search", "arguments": {"q": "test"}})
        assert tc.id == "tc2"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}

    def test_from_dict_no_arguments(self) -> None:
        tc = ToolCall.from_dict({"id": "tc3", "name": "noop"})
        assert tc.arguments == {}


# ---------------------------------------------------------------------------
# LLMMessage serialisation
# ---------------------------------------------------------------------------


class TestLLMMessage:
    def test_to_openai_dict_basic(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        d = msg.to_openai_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_openai_dict_with_tool_calls(self) -> None:
        msg = LLMMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc1", name="fn", arguments={"a": 1})],
        )
        d = msg.to_openai_dict()
        assert d["tool_calls"][0]["function"]["name"] == "fn"
        assert json.loads(d["tool_calls"][0]["function"]["arguments"]) == {"a": 1}

    def test_to_openai_dict_with_tool_call_id(self) -> None:
        msg = LLMMessage(role="tool", content="result", tool_call_id="tc1")
        d = msg.to_openai_dict()
        assert d["tool_call_id"] == "tc1"

    def test_to_anthropic_dict_user(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        d = msg.to_anthropic_dict()
        assert d == {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    def test_to_anthropic_dict_tool_result(self) -> None:
        msg = LLMMessage(role="tool", content="result data", tool_call_id="tu1")
        d = msg.to_anthropic_dict()
        assert d["role"] == "user"
        assert d["content"][0]["type"] == "tool_result"
        assert d["content"][0]["tool_use_id"] == "tu1"

    def test_to_anthropic_dict_with_tool_calls(self) -> None:
        msg = LLMMessage(
            role="assistant",
            content="Let me check",
            tool_calls=[ToolCall(id="tu1", name="search", arguments={"q": "test"})],
        )
        d = msg.to_anthropic_dict()
        assert d["role"] == "assistant"
        blocks = d["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "search"


# ---------------------------------------------------------------------------
# Tool format converters
# ---------------------------------------------------------------------------


class TestToolConverters:
    def test_to_openai_tools_passthrough(self) -> None:
        tools = [{"type": "function", "function": {"name": "fn", "parameters": {}}}]
        assert _to_openai_tools(tools) == tools

    def test_to_openai_tools_convert(self) -> None:
        tools = [{"name": "fn", "description": "desc", "parameters": {"type": "object"}}]
        result = _to_openai_tools(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "fn"

    def test_to_anthropic_tools_from_openai_format(self) -> None:
        tools = [{"type": "function", "function": {"name": "fn", "description": "d", "parameters": {}}}]
        result = _to_anthropic_tools(tools)
        assert result[0]["name"] == "fn"
        assert "input_schema" in result[0]

    def test_to_anthropic_tools_passthrough(self) -> None:
        tools = [{"name": "fn", "description": "d", "input_schema": {}}]
        assert _to_anthropic_tools(tools) == tools

    def test_to_ollama_tools_convert(self) -> None:
        tools = [{"name": "fn", "description": "d", "parameters": {}}]
        result = _to_ollama_tools(tools)
        assert result[0]["type"] == "function"


# ---------------------------------------------------------------------------
# extract_system helper
# ---------------------------------------------------------------------------


class TestExtractSystem:
    def test_separates_system(self) -> None:
        msgs = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hi"),
        ]
        system, rest = _extract_system(msgs)
        assert system == "You are helpful"
        assert len(rest) == 1
        assert rest[0].role == "user"

    def test_no_system(self) -> None:
        msgs = [LLMMessage(role="user", content="Hi")]
        system, rest = _extract_system(msgs)
        assert system == ""
        assert len(rest) == 1

    def test_multiple_system(self) -> None:
        msgs = [
            LLMMessage(role="system", content="A"),
            LLMMessage(role="system", content="B"),
            LLMMessage(role="user", content="Hi"),
        ]
        system, rest = _extract_system(msgs)
        assert "A" in system and "B" in system
        assert len(rest) == 1


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    @pytest.fixture
    def provider(self) -> OllamaProvider:
        return OllamaProvider(base_url="http://localhost:11434", model="llama3.1")

    async def test_chat(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {"content": "Hello back!", "tool_calls": []},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Hello")]
        resp = await provider.chat(msgs)
        assert resp.content == "Hello back!"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        mock_client.chat.assert_called_once()

    async def test_chat_with_tool_calls(self, provider: OllamaProvider) -> None:
        mock_client = AsyncMock()
        mock_client.chat.return_value = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
            },
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Weather?")]
        tools = [{"name": "get_weather", "parameters": {"type": "object"}}]
        resp = await provider.chat(msgs, tools=tools)
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "Berlin"}

    async def test_count_tokens(self, provider: OllamaProvider) -> None:
        count = await provider.count_tokens("hello world test")
        assert count == len("hello world test") // 4


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-key", model="gpt-4o")

    async def test_chat(self, provider: OpenAIProvider) -> None:
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hi there!", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            model="gpt-4o",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Hello")]
        resp = await provider.chat(msgs)
        assert resp.content == "Hi there!"
        assert resp.input_tokens == 12
        assert resp.output_tokens == 4

    async def test_chat_with_tool_calls(self, provider: OpenAIProvider) -> None:
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="search", arguments='{"q": "test"}'),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=6),
            model="gpt-4o",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Search")]
        resp = await provider.chat(msgs, tools=[{"name": "search", "parameters": {}}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "test"}

    async def test_count_tokens_fallback(self, provider: OpenAIProvider) -> None:
        count = await provider.count_tokens("test string here")
        # Falls back to len//4 if tiktoken not available
        assert isinstance(count, int)
        assert count > 0


# ---------------------------------------------------------------------------
# ClaudeProvider
# ---------------------------------------------------------------------------


class TestClaudeProvider:
    @pytest.fixture
    def provider(self) -> ClaudeProvider:
        return ClaudeProvider(api_key="test-key", model="claude-sonnet-4-20250514")

    async def test_chat(self, provider: ClaudeProvider) -> None:
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello from Claude!")],
            usage=SimpleNamespace(input_tokens=15, output_tokens=7),
            model="claude-sonnet-4-20250514",
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Hello")]
        resp = await provider.chat(msgs)
        assert resp.content == "Hello from Claude!"
        assert resp.input_tokens == 15
        assert resp.output_tokens == 7

    async def test_chat_with_tool_use(self, provider: ClaudeProvider) -> None:
        mock_response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me check"),
                SimpleNamespace(type="tool_use", id="tu_1", name="get_weather", input={"city": "Berlin"}),
            ],
            usage=SimpleNamespace(input_tokens=20, output_tokens=10),
            model="claude-sonnet-4-20250514",
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Weather?")]
        tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]
        resp = await provider.chat(msgs, tools=tools)
        assert resp.content == "Let me check"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "Berlin"}

    async def test_chat_with_system_message(self, provider: ClaudeProvider) -> None:
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            model="claude-sonnet-4-20250514",
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hi"),
        ]
        await provider.chat(msgs)
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
        # system message should NOT be in the messages list
        for m in call_kwargs["messages"]:
            assert m["role"] != "system"

    async def test_count_tokens_fallback(self, provider: ClaudeProvider) -> None:
        # Mock client that raises so fallback is used
        mock_client = AsyncMock()
        mock_client.count_tokens = AsyncMock(side_effect=Exception("not available"))
        provider._client = mock_client
        count = await provider.count_tokens("test")
        assert count == 1  # len("test") // 4


# ---------------------------------------------------------------------------
# AzureOpenAIProvider
# ---------------------------------------------------------------------------


class TestAzureOpenAIProvider:
    @pytest.fixture
    def provider(self) -> AzureOpenAIProvider:
        return AzureOpenAIProvider(
            api_key="azure-key",
            endpoint="https://myresource.openai.azure.com",
            model="gpt-4o",
            api_version="2024-10-21",
        )

    async def test_chat(self, provider: AzureOpenAIProvider) -> None:
        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Azure response", tool_calls=None)),
            ],
            usage=SimpleNamespace(prompt_tokens=8, completion_tokens=3),
            model="gpt-4o",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="Hello")]
        resp = await provider.chat(msgs)
        assert resp.content == "Azure response"
        assert resp.input_tokens == 8


# ---------------------------------------------------------------------------
# LLMRegistry
# ---------------------------------------------------------------------------


class TestLLMRegistry:
    def test_register_and_get(self) -> None:
        registry = LLMRegistry()
        provider = OllamaProvider()
        registry.register("ollama", provider, default=True)
        assert registry.get("ollama") is provider
        assert registry.get() is provider  # default

    def test_get_missing_raises(self) -> None:
        registry = LLMRegistry()
        with pytest.raises(ValueError, match="not found"):
            registry.get("nonexistent")

    def test_from_config(self) -> None:
        providers = {
            "ollama": LLMProviderConfig(type=LLMProviderType.OLLAMA, default_model="llama3.1", enabled=True),
            "openai": LLMProviderConfig(type=LLMProviderType.OPENAI, api_key="k", default_model="gpt-4o", enabled=False),
        }
        registry = LLMRegistry.from_config(providers, default="ollama")
        assert isinstance(registry.get("ollama"), OllamaProvider)
        with pytest.raises(ValueError):
            registry.get("openai")  # disabled

    def test_from_config_all_providers(self) -> None:
        providers = {
            "ollama": LLMProviderConfig(type=LLMProviderType.OLLAMA, default_model="llama3.1", base_url="http://localhost:11434"),
            "openai": LLMProviderConfig(type=LLMProviderType.OPENAI, api_key="k", default_model="gpt-4o"),
            "claude": LLMProviderConfig(type=LLMProviderType.CLAUDE, api_key="k", default_model="claude-sonnet-4-20250514"),
            "azure": LLMProviderConfig(
                type=LLMProviderType.AZURE_OPENAI,
                api_key="k",
                base_url="https://x.openai.azure.com",
                default_model="gpt-4o",
                extra={"api_version": "2024-10-21"},
            ),
        }
        registry = LLMRegistry.from_config(providers, default="ollama")
        assert isinstance(registry.get("ollama"), OllamaProvider)
        assert isinstance(registry.get("openai"), OpenAIProvider)
        assert isinstance(registry.get("claude"), ClaudeProvider)
        assert isinstance(registry.get("azure"), AzureOpenAIProvider)


# ---------------------------------------------------------------------------
# Multi-model and deployment_map tests
# ---------------------------------------------------------------------------


class TestModelOverride:
    """Test that model= parameter overrides the default model in each provider."""

    async def test_ollama_model_override(self) -> None:
        provider = OllamaProvider(model="llama3.1")
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value={"message": {"content": "ok"}, "prompt_eval_count": 0, "eval_count": 0})
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        resp = await provider.chat(msgs, model="qwen2.5")
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["model"] == "qwen2.5"
        assert resp.model == "qwen2.5"

    async def test_openai_model_override(self) -> None:
        provider = OpenAIProvider(api_key="k", model="gpt-4o")
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="gpt-4o-mini",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        await provider.chat(msgs, model="gpt-4o-mini")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"

    async def test_claude_model_override(self) -> None:
        provider = ClaudeProvider(api_key="k", model="claude-sonnet-4-20250514")
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            model="claude-haiku-4-5-20251001",
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        await provider.chat(msgs, model="claude-haiku-4-5-20251001")
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


class TestAzureDeploymentMap:
    """Test Azure deployment_name resolution."""

    async def test_deployment_map_resolves(self) -> None:
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint="https://x.openai.azure.com",
            model="gpt-4o",
            deployment_map={"gpt-4o": "my-gpt4o-deploy", "gpt-4o-mini": "my-mini-deploy"},
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="my-gpt4o-deploy",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        # Default model should resolve to deployment name
        await provider.chat(msgs)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-gpt4o-deploy"

    async def test_deployment_map_with_model_override(self) -> None:
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint="https://x.openai.azure.com",
            model="gpt-4o",
            deployment_map={"gpt-4o": "my-gpt4o-deploy", "gpt-4o-mini": "my-mini-deploy"},
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="my-mini-deploy",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        await provider.chat(msgs, model="gpt-4o-mini")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "my-mini-deploy"

    async def test_deployment_map_fallback_no_mapping(self) -> None:
        provider = AzureOpenAIProvider(
            api_key="k",
            endpoint="https://x.openai.azure.com",
            model="gpt-4o",
            deployment_map={},
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="gpt-4o",
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        provider._client = mock_client

        msgs = [LLMMessage(role="user", content="hi")]
        await provider.chat(msgs)
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"  # no mapping, uses model name directly

    def test_registry_builds_deployment_map(self) -> None:
        providers = {
            "azure": LLMProviderConfig(
                type=LLMProviderType.AZURE_OPENAI,
                api_key="k",
                base_url="https://x.openai.azure.com",
                default_model="gpt-4o",
                api_version="2024-10-21",
                models=[
                    {"name": "gpt-4o", "deployment_name": "my-gpt4o"},
                    {"name": "gpt-4o-mini", "deployment_name": "my-mini"},
                ],
            ),
        }
        registry = LLMRegistry.from_config(providers, default="azure")
        azure_provider = registry.get("azure")
        assert isinstance(azure_provider, AzureOpenAIProvider)
        assert azure_provider._deployment_map == {"gpt-4o": "my-gpt4o", "gpt-4o-mini": "my-mini"}
