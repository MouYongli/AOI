"""OpenAI LLM provider."""

from __future__ import annotations

from typing import Any

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = structlog.get_logger()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str | None = None) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # TODO: implement using openai Python SDK
        logger.info("openai.chat", model=self._model)
        return LLMResponse(content="[OpenAI placeholder]", model=self._model)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4
