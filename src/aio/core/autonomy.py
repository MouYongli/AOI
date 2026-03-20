"""AutonomyManager: three-level autonomy control for tool execution."""

from __future__ import annotations

from enum import Enum

import structlog

from aio.config.settings import AutonomyLevel

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Confirmation matrix: (autonomy_level, risk_level) → requires_confirmation
_CONFIRMATION_MATRIX: dict[tuple[AutonomyLevel, RiskLevel], bool] = {
    # full-auto: only CRITICAL requires confirmation
    (AutonomyLevel.FULL_AUTO, RiskLevel.LOW): False,
    (AutonomyLevel.FULL_AUTO, RiskLevel.MEDIUM): False,
    (AutonomyLevel.FULL_AUTO, RiskLevel.HIGH): False,
    (AutonomyLevel.FULL_AUTO, RiskLevel.CRITICAL): True,
    # semi-auto: HIGH+ requires confirmation
    (AutonomyLevel.SEMI_AUTO, RiskLevel.LOW): False,
    (AutonomyLevel.SEMI_AUTO, RiskLevel.MEDIUM): False,
    (AutonomyLevel.SEMI_AUTO, RiskLevel.HIGH): True,
    (AutonomyLevel.SEMI_AUTO, RiskLevel.CRITICAL): True,
    # user-confirm: everything requires confirmation
    (AutonomyLevel.USER_CONFIRM, RiskLevel.LOW): True,
    (AutonomyLevel.USER_CONFIRM, RiskLevel.MEDIUM): True,
    (AutonomyLevel.USER_CONFIRM, RiskLevel.HIGH): True,
    (AutonomyLevel.USER_CONFIRM, RiskLevel.CRITICAL): True,
}


class AutonomyManager:
    """Manage autonomy levels with three-tier configuration priority.

    Priority: conversation-level > agent template-level > system default.
    """

    def __init__(self, default_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO) -> None:
        self._system_default = default_level
        self._session_overrides: dict[str, AutonomyLevel] = {}

    def get_effective_level(
        self,
        session_id: str,
        template_level: AutonomyLevel | None = None,
    ) -> AutonomyLevel:
        """Resolve effective autonomy level using priority chain."""
        if session_id in self._session_overrides:
            return self._session_overrides[session_id]
        if template_level is not None:
            return template_level
        return self._system_default

    def set_session_override(self, session_id: str, level: AutonomyLevel) -> None:
        self._session_overrides[session_id] = level

    def clear_session_override(self, session_id: str) -> None:
        self._session_overrides.pop(session_id, None)

    def should_confirm(self, tool_name: str, risk: RiskLevel, autonomy: AutonomyLevel) -> bool:
        """Determine if a tool call requires user confirmation."""
        return _CONFIRMATION_MATRIX.get((autonomy, risk), True)
