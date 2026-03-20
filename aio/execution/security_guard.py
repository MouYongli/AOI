"""SecurityGuard: whitelist/blacklist + risk assessment for tool execution."""

from __future__ import annotations

from pathlib import Path

import structlog

from aio.core.autonomy import RiskLevel

logger = structlog.get_logger()


class SecurityGuard:
    """Evaluate tool calls against security rules before execution."""

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        blocked_commands: list[str] | None = None,
    ) -> None:
        self._allowed_dirs = [Path(d).expanduser().resolve() for d in (allowed_directories or ["~"])]
        self._blocked_commands = blocked_commands or []

    def check_path(self, path: str) -> bool:
        """Check if a file path is within allowed directories."""
        resolved = Path(path).resolve()
        return any(self._is_subpath(resolved, allowed) for allowed in self._allowed_dirs)

    def check_command(self, command: str) -> bool:
        """Check if a command is not in the blocklist."""
        cmd_lower = command.lower().strip()
        return not any(blocked in cmd_lower for blocked in self._blocked_commands)

    def assess_risk(self, tool_name: str, arguments: dict) -> RiskLevel:
        """Assess the risk level of a tool call."""
        # File write/delete operations
        if tool_name in ("write_file", "delete_file", "move_file"):
            return RiskLevel.HIGH
        # Command execution
        if tool_name in ("run_command", "execute"):
            return RiskLevel.HIGH
        # File read, search
        if tool_name in ("read_file", "list_directory", "search"):
            return RiskLevel.LOW
        # Web browsing
        if tool_name in ("navigate", "click", "screenshot"):
            return RiskLevel.MEDIUM
        return RiskLevel.MEDIUM

    @staticmethod
    def _is_subpath(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
