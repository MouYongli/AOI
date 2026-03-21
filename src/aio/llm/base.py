"""Base LLM provider interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ToolCall:
    """Normalized tool/function call representation across providers."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCall:
        return cls(id=d["id"], name=d["name"], arguments=d.get("arguments", {}))


@dataclass
class LLMMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    # --- OpenAI-compatible serialisation helpers ---

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert to OpenAI chat-completion message format."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Convert to Anthropic messages-API format."""
        if self.role == "tool":
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self.tool_call_id or "",
                        "content": self.content,
                    }
                ],
            }
        blocks: list[dict[str, Any]] = []
        if self.content:
            blocks.append({"type": "text", "text": self.content})
        for tc in self.tool_calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
        return {"role": self.role, "content": blocks or self.content}


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


class LLMProvider(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
    ) -> LLMResponse:
        """Send a chat completion request."""

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        response = await self.chat(messages, tools, stream=False)
        yield response.content

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for the given text."""
