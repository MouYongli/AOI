"""Claude (Anthropic) LLM provider."""

from __future__ import annotations

from typing import Any, AsyncIterator

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

logger = structlog.get_logger()


def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic tool definitions to Anthropic tool format."""
    result: list[dict[str, Any]] = []
    for t in tools:
        if "input_schema" in t:
            result.append(t)
        elif "function" in t:
            func = t["function"]
            result.append({"name": func["name"], "description": func.get("description", ""), "input_schema": func.get("parameters", {})})
        else:
            result.append({"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("parameters", {})})
    return result


def _extract_system(messages: list[LLMMessage]) -> tuple[str, list[LLMMessage]]:
    """Separate the system message (Anthropic requires it as a top-level param)."""
    system = ""
    rest: list[LLMMessage] = []
    for m in messages:
        if m.role == "system":
            system = m.content if not system else f"{system}\n\n{m.content}"
        else:
            rest.append(m)
    return system, rest


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider using the official anthropic Python SDK."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", max_tokens: int = 4096) -> None:
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        system, rest = _extract_system(messages)
        api_msgs = [m.to_anthropic_dict() for m in rest]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        logger.info("claude.chat", model=self._model, stream=stream)

        if stream:
            return await self._chat_stream_collect(kwargs)

        response = await client.messages.create(**kwargs)

        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input or {}))

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )

    async def _chat_stream_collect(self, kwargs: dict[str, Any]) -> LLMResponse:
        """Call streaming API and collect into a single LLMResponse."""
        client = self._get_client()
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        current_tool: dict[str, Any] | None = None
        input_tokens = 0
        output_tokens = 0

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {"id": event.content_block.id, "name": event.content_block.name, "arguments": ""}
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        content_parts.append(event.delta.text)
                    elif event.delta.type == "input_json_delta" and current_tool is not None:
                        current_tool["arguments"] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        import json

                        try:
                            args = json.loads(current_tool["arguments"]) if current_tool["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append(ToolCall(id=current_tool["id"], name=current_tool["name"], arguments=args))
                        current_tool = None
                elif event.type == "message_delta":
                    if hasattr(event.usage, "output_tokens"):
                        output_tokens = event.usage.output_tokens
                elif event.type == "message_start":
                    if hasattr(event.message, "usage"):
                        input_tokens = event.message.usage.input_tokens

        return LLMResponse(
            content="".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._model,
        )

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        system, rest = _extract_system(messages)
        api_msgs = [m.to_anthropic_dict() for m in rest]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": api_msgs,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        logger.info("claude.chat_stream", model=self._model)
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def count_tokens(self, text: str) -> int:
        try:
            client = self._get_client()
            response = await client.count_tokens(
                model=self._model,
                messages=[{"role": "user", "content": text}],
            )
            return response.input_tokens
        except Exception:
            return len(text) // 4
