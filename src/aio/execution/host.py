"""Host execution environment: run commands on the host machine."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from aio.execution.base import ExecutionEnvironment, ExecutionResult

logger = structlog.get_logger()


class HostExecutionEnvironment(ExecutionEnvironment):
    """Execute commands directly on the host machine."""

    async def execute(self, command: str, **kwargs: Any) -> ExecutionResult:
        logger.info("host.execute", command=command[:100])
        timeout = kwargs.get("timeout", 30)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecutionResult(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                exit_code=proc.returncode or 0,
                is_error=(proc.returncode or 0) != 0,
            )
        except asyncio.TimeoutError:
            return ExecutionResult(stderr="Command timed out", exit_code=1, is_error=True)

    async def health_check(self) -> bool:
        return True
