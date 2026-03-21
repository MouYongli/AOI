"""Ollama LLM provider."""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall

logger = structlog.get_logger()


def _to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic tool definitions to Ollama format (OpenAI-compatible)."""
    ollama_tools: list[dict[str, Any]] = []
    for t in tools:
        if "function" in t:
            ollama_tools.append(t)
        else:
            ollama_tools.append(
                {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
            )
    return ollama_tools


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider using the official ollama Python SDK."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1") -> None:
        self._base_url = base_url
        self._model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from ollama import AsyncClient

            self._client = AsyncClient(host=self._base_url)
        return self._client

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        ollama_msgs = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {"model": self._model, "messages": ollama_msgs}
        if tools:
            kwargs["tools"] = _to_ollama_tools(tools)

        logger.info("ollama.chat", model=self._model)
        response = await client.chat(**kwargs)

        content = response.get("message", {}).get("content", "")
        tool_calls: list[ToolCall] = []
        for tc in response.get("message", {}).get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(id=str(uuid.uuid4()), name=func.get("name", ""), arguments=args))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            model=self._model,
        )

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        client = self._get_client()
        ollama_msgs = [{"role": m.role, "content": m.content} for m in messages]

        logger.info("ollama.chat_stream", model=self._model)
        stream = await client.chat(model=self._model, messages=ollama_msgs, stream=True)
        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate; Ollama doesn't expose tokenizer
