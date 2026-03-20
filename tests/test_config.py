"""Tests for configuration loading."""

from aio.config.settings import AIOConfig, AutonomyLevel, ExecutionEnv, load_config


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
