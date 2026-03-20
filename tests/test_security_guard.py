"""Tests for SecurityGuard."""

from aio.core.autonomy import RiskLevel
from aio.execution.security_guard import SecurityGuard


def test_check_command_allowed() -> None:
    guard = SecurityGuard(blocked_commands=["rm -rf /"])
    assert guard.check_command("ls -la") is True


def test_check_command_blocked() -> None:
    guard = SecurityGuard(blocked_commands=["rm -rf /"])
    assert guard.check_command("rm -rf /") is False


def test_assess_risk_read_file() -> None:
    guard = SecurityGuard()
    assert guard.assess_risk("read_file", {}) == RiskLevel.LOW


def test_assess_risk_write_file() -> None:
    guard = SecurityGuard()
    assert guard.assess_risk("write_file", {}) == RiskLevel.HIGH


def test_assess_risk_navigate() -> None:
    guard = SecurityGuard()
    assert guard.assess_risk("navigate", {}) == RiskLevel.MEDIUM
