"""Azure OpenAI LLM provider."""

from __future__ import annotations

from typing import Any

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = structlog.get_logger()


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI API provider."""

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

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # TODO: implement using openai Python SDK with azure config
        logger.info("azure_openai.chat", model=self._model)
        return LLMResponse(content="[Azure OpenAI placeholder]", model=self._model)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4
