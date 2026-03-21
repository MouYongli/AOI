"""LLMRegistry: manage multiple LLM providers, auto-convert tool formats."""

from __future__ import annotations

import structlog

from aio.config.settings import LLMProviderConfig, LLMProviderType
from aio.llm.azure_openai import AzureOpenAIProvider
from aio.llm.base import LLMProvider
from aio.llm.claude_provider import ClaudeProvider
from aio.llm.ollama import OllamaProvider
from aio.llm.openai_provider import OpenAIProvider

logger = structlog.get_logger()


class LLMRegistry:
    """Registry managing all configured LLM providers."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self._default: str | None = None

    def register(self, name: str, provider: LLMProvider, default: bool = False) -> None:
        self._providers[name] = provider
        if default or self._default is None:
            self._default = name

    def get(self, name: str | None = None) -> LLMProvider:
        key = name or self._default
        if not self._providers:
            raise ValueError("No LLM providers are enabled. Check llm_providers config and ensure at least one has enabled: true")
        if key is None or key not in self._providers:
            raise ValueError(f"LLM provider '{key}' not found. Available: {list(self._providers.keys())}")
        return self._providers[key]

    @property
    def available_providers(self) -> list[str]:
        return list(self._providers.keys())

    @classmethod
    def from_config(cls, providers: dict[str, LLMProviderConfig], default: str) -> LLMRegistry:
        registry = cls()
        for name, cfg in providers.items():
            if not cfg.enabled:
                logger.info("llm_provider.skipped", name=name, reason="disabled")
                continue
            provider = _build_provider(cfg)
            registry.register(name, provider, default=(name == default))

        if not registry._providers:
            logger.warning("llm_registry.no_providers", msg="No LLM providers enabled")
        elif registry._default not in registry._providers:
            # default_llm_provider was disabled; fall back to first enabled
            fallback = next(iter(registry._providers))
            logger.warning(
                "llm_registry.default_disabled",
                requested=default,
                fallback=fallback,
            )
            registry._default = fallback

        return registry


def _build_provider(cfg: LLMProviderConfig) -> LLMProvider:
    match cfg.type:
        case LLMProviderType.OLLAMA:
            return OllamaProvider(base_url=cfg.base_url or "http://localhost:11434", model=cfg.default_model)
        case LLMProviderType.OPENAI:
            return OpenAIProvider(api_key=cfg.api_key or "", model=cfg.default_model, base_url=cfg.base_url)
        case LLMProviderType.CLAUDE:
            return ClaudeProvider(api_key=cfg.api_key or "", model=cfg.default_model)
        case LLMProviderType.AZURE_OPENAI:
            deployment_map: dict[str, str] = {}
            for mc in cfg.models:
                if mc.deployment_name:
                    deployment_map[mc.name] = mc.deployment_name
            return AzureOpenAIProvider(
                api_key=cfg.api_key or "",
                endpoint=cfg.base_url or "",
                model=cfg.default_model,
                api_version=cfg.api_version or cfg.extra.get("api_version", "2024-10-21"),
                deployment_map=deployment_map,
            )
        case _:
            raise ValueError(f"Unknown LLM provider type: {cfg.type}")
