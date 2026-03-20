"""Tests for AutonomyManager."""

from aio.config.settings import AutonomyLevel
from aio.core.autonomy import AutonomyManager, RiskLevel


def test_default_level() -> None:
    mgr = AutonomyManager(default_level=AutonomyLevel.SEMI_AUTO)
    assert mgr.get_effective_level("session-1") == AutonomyLevel.SEMI_AUTO


def test_session_override() -> None:
    mgr = AutonomyManager()
    mgr.set_session_override("s1", AutonomyLevel.FULL_AUTO)
    assert mgr.get_effective_level("s1") == AutonomyLevel.FULL_AUTO


def test_session_override_priority_over_template() -> None:
    mgr = AutonomyManager()
    mgr.set_session_override("s1", AutonomyLevel.USER_CONFIRM)
    assert mgr.get_effective_level("s1", template_level=AutonomyLevel.FULL_AUTO) == AutonomyLevel.USER_CONFIRM


def test_template_level_over_default() -> None:
    mgr = AutonomyManager(default_level=AutonomyLevel.SEMI_AUTO)
    assert mgr.get_effective_level("s1", template_level=AutonomyLevel.FULL_AUTO) == AutonomyLevel.FULL_AUTO


def test_should_confirm_semi_auto_high() -> None:
    mgr = AutonomyManager()
    assert mgr.should_confirm("write_file", RiskLevel.HIGH, AutonomyLevel.SEMI_AUTO) is True


def test_should_confirm_semi_auto_low() -> None:
    mgr = AutonomyManager()
    assert mgr.should_confirm("read_file", RiskLevel.LOW, AutonomyLevel.SEMI_AUTO) is False


def test_should_confirm_full_auto_critical() -> None:
    mgr = AutonomyManager()
    assert mgr.should_confirm("rm_rf", RiskLevel.CRITICAL, AutonomyLevel.FULL_AUTO) is True


def test_should_confirm_full_auto_high() -> None:
    mgr = AutonomyManager()
    assert mgr.should_confirm("write_file", RiskLevel.HIGH, AutonomyLevel.FULL_AUTO) is False
