"""Claude (Anthropic) LLM provider."""

from __future__ import annotations

from typing import Any

import structlog

from aio.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = structlog.get_logger()


class ClaudeProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._api_key = api_key
        self._model = model

    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        # TODO: implement using anthropic Python SDK
        logger.info("claude.chat", model=self._model)
        return LLMResponse(content="[Claude placeholder]", model=self._model)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4
