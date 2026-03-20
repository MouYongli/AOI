"""SkillEngine: progressive disclosure of tools based on context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class Skill:
    name: str
    description: str
    tools: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)


class SkillEngine:
    """Evaluate context and dynamically adjust available tool set.

    Implements Anthropic's Progressive Disclosure pattern:
    expose only relevant tools to reduce confusion and improve accuracy.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register_skill(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def evaluate(self, context: dict[str, Any], available_skills: list[str]) -> list[str]:
        """Return list of tool names that should be active for the current context."""
        active_tools: list[str] = []
        for skill_name in available_skills:
            skill = self._skills.get(skill_name)
            if skill:
                active_tools.extend(skill.tools)
        return active_tools
