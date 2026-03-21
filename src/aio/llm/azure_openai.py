"""Azure OpenAI LLM provider."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall
from aio.llm.openai_provider import _to_openai_tools

logger = structlog.get_logger()


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider using the official openai Python SDK with Azure config."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        model: str = "gpt-4o",
        api_version: str = "2024-10-21",
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._model = model
        self._api_version = api_version
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncAzureOpenAI

            self._client = AsyncAzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version=self._api_version,
            )
        return self._client

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        oai_msgs = [m.to_openai_dict() for m in messages]

        kwargs: dict[str, Any] = {"model": self._model, "messages": oai_msgs}
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        logger.info("azure_openai.chat", model=self._model, stream=stream)

        if stream:
            return await self._chat_stream_collect(kwargs)

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    args_dict = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args_dict))

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
        )

    async def _chat_stream_collect(self, kwargs: dict[str, Any]) -> LLMResponse:
        """Call streaming API and collect into a single LLMResponse."""
        client = self._get_client()
        kwargs["stream"] = True
        stream = await client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        tc_map: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": tc_delta.id or "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["arguments"] += tc_delta.function.arguments

        tool_calls: list[ToolCall] = []
        for _, tc_data in sorted(tc_map.items()):
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))

        return LLMResponse(content="".join(content_parts), tool_calls=tool_calls, model=self._model)

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        oai_msgs = [m.to_openai_dict() for m in messages]

        kwargs: dict[str, Any] = {"model": self._model, "messages": oai_msgs, "stream": True}
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)

        logger.info("azure_openai.chat_stream", model=self._model)
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self._model)
            return len(enc.encode(text))
        except Exception:
            return len(text) // 4
