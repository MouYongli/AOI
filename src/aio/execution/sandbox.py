"""OpenSandbox execution environment: run commands in an isolated container."""

from __future__ import annotations

from typing import Any

import structlog

from aio.execution.base import ExecutionEnvironment, ExecutionResult

logger = structlog.get_logger()


class SandboxExecutionEnvironment(ExecutionEnvironment):
    """Execute commands in an OpenSandbox container."""

    def __init__(self, sandbox_url: str = "http://localhost:8080") -> None:
        self._sandbox_url = sandbox_url

    async def execute(self, command: str, **kwargs: Any) -> ExecutionResult:
        logger.info("sandbox.execute", command=command[:100])
        # TODO: implement via OpenSandbox API
        return ExecutionResult(stdout="[Sandbox placeholder]", exit_code=0)

    async def health_check(self) -> bool:
        # TODO: check OpenSandbox availability
        return False
