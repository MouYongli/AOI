"""Tests for configuration loading."""

from aio.config.settings import (
    AIOConfig,
    AutonomyLevel,
    ExecutionEnv,
    LLMProviderConfig,
    LLMProviderType,
    ModelConfig,
    load_config,
)


def test_default_config() -> None:
    """Default config should have sensible defaults."""
    config = AIOConfig()
    assert config.default_autonomy_level == AutonomyLevel.SEMI_AUTO
    assert config.default_execution_env == ExecutionEnv.HOST
    assert config.max_concurrent_sessions == 3


def test_load_config_missing_file() -> None:
    """Loading from a non-existent file should return defaults."""
    config = load_config("nonexistent.yaml")
    assert isinstance(config, AIOConfig)


# ---------------------------------------------------------------------------
# ModelConfig and LLMProviderConfig.models tests
# ---------------------------------------------------------------------------


def test_models_string_list() -> None:
    """Plain strings should be normalized to ModelConfig objects."""
    cfg = LLMProviderConfig(
        type=LLMProviderType.OLLAMA,
        default_model="llama3.1",
        models=["llama3.1", "qwen2.5"],
    )
    assert len(cfg.models) == 2
    assert all(isinstance(m, ModelConfig) for m in cfg.models)
    assert cfg.models[0].name == "llama3.1"
    assert cfg.models[1].name == "qwen2.5"
    assert cfg.models[0].deployment_name is None


def test_models_dict_list() -> None:
    """Dicts with deployment_name should parse correctly."""
    cfg = LLMProviderConfig(
        type=LLMProviderType.AZURE_OPENAI,
        default_model="gpt-4o",
        api_version="2024-10-21",
        models=[
            {"name": "gpt-4o", "deployment_name": "my-deploy"},
            {"name": "gpt-4o-mini", "deployment_name": "my-mini"},
        ],
    )
    assert cfg.models[0].deployment_name == "my-deploy"
    assert cfg.models[1].name == "gpt-4o-mini"


def test_models_empty_default() -> None:
    """Empty models list is the default — backward compatible."""
    cfg = LLMProviderConfig(
        type=LLMProviderType.OPENAI,
        default_model="gpt-4o",
    )
    assert cfg.models == []


def test_get_model_config() -> None:
    """get_model_config should look up by name."""
    cfg = LLMProviderConfig(
        type=LLMProviderType.AZURE_OPENAI,
        default_model="gpt-4o",
        models=[
            {"name": "gpt-4o", "deployment_name": "deploy-4o"},
            {"name": "gpt-4o-mini"},
        ],
    )
    mc = cfg.get_model_config("gpt-4o")
    assert mc is not None
    assert mc.deployment_name == "deploy-4o"

    mc_mini = cfg.get_model_config("gpt-4o-mini")
    assert mc_mini is not None
    assert mc_mini.deployment_name is None

    assert cfg.get_model_config("nonexistent") is None


def test_api_version_field() -> None:
    """api_version should be a first-class field."""
    cfg = LLMProviderConfig(
        type=LLMProviderType.AZURE_OPENAI,
        default_model="gpt-4o",
        api_version="2024-10-21",
    )
    assert cfg.api_version == "2024-10-21"
