"""Ollama LLM provider."""

from __future__ import annotations

from typing import Any

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = structlog.get_logger()


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.1") -> None:
        self._base_url = base_url
        self._model = model

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # TODO: implement using ollama Python SDK
        logger.info("ollama.chat", model=self._model)
        return LLMResponse(content="[Ollama placeholder]", model=self._model)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate
