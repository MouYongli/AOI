"""Configuration schema using Pydantic models, loaded from YAML."""

from __future__ import annotations

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class AutonomyLevel(str, Enum):
    FULL_AUTO = "full-auto"
    SEMI_AUTO = "semi-auto"
    USER_CONFIRM = "user-confirm"


class ExecutionEnv(str, Enum):
    HOST = "host"
    SANDBOX = "sandbox"


class LLMProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    CLAUDE = "claude"
    AZURE_OPENAI = "azure_openai"


class LLMProviderConfig(BaseModel):
    type: LLMProviderType
    base_url: str | None = None
    api_key: str | None = None
    default_model: str
    enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    allowed_user_ids: list[int] = Field(default_factory=list)

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def parse_comma_separated_ids(cls, v: Any) -> list[int]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        return v


class CLIConfig(BaseModel):
    enabled: bool = True


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class SecurityConfig(BaseModel):
    allowed_directories: list[str] = Field(default_factory=lambda: ["~"])
    blocked_commands: list[str] = Field(
        default_factory=lambda: ["rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:"]
    )
    autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO


class AgentTemplateConfig(BaseModel):
    name: str
    description: str
    system_prompt: str = ""
    llm_provider: str | None = None
    llm_model: str | None = None
    mcp_servers: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    execution_env: ExecutionEnv | None = None
    autonomy_level: AutonomyLevel | None = None
    max_react_loops: int = 10


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://aio:aio@localhost:5432/aio"


class AIOConfig(BaseModel):
    """Root configuration for AIO."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    default_llm_provider: str = "ollama"
    llm_providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    agent_templates: list[AgentTemplateConfig] = Field(default_factory=list)
    default_execution_env: ExecutionEnv = ExecutionEnv.HOST
    default_autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO
    max_concurrent_sessions: int = 3


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} references in strings using os.environ."""
    if isinstance(obj, str):
        def _replace(m: re.Match) -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_VAR_PATTERN.sub(_replace, obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(path: str | Path | None = None) -> AIOConfig:
    """Load configuration from YAML file.

    Resolution order for config path:
    1. Explicit ``path`` argument
    2. ``AIO_CONFIG_PATH`` environment variable
    3. Default ``config/config.yaml``
    """
    if path is None:
        path = os.environ.get("AIO_CONFIG_PATH", "config/config.yaml")
    config_path = Path(path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        data = _expand_env_vars(data)
        return AIOConfig.model_validate(data)
    return AIOConfig()
