"""SequentialRunner: V1 workflow execution — run steps one by one."""

from __future__ import annotations

from typing import Any

import structlog

from aio.workflow.base import Workflow, WorkflowRunner, WorkflowStatus

logger = structlog.get_logger()


class SequentialRunner(WorkflowRunner):
    """Execute workflow steps sequentially (Phase 1)."""

    def __init__(self) -> None:
        self._workflows: dict[str, Workflow] = {}

    async def run(self, workflow: Workflow, progress_cb: Any = None) -> dict[str, Any]:
        workflow.status = WorkflowStatus.RUNNING
        self._workflows[workflow.id] = workflow
        results: dict[str, Any] = {}

        for step in workflow.steps:
            step.status = WorkflowStatus.RUNNING
            logger.info("sequential.step", step_id=step.id, task=step.task)

            try:
                # TODO: spawn AgentInstance and execute
                step.result = f"[Step {step.id}] completed (placeholder)"
                step.status = WorkflowStatus.COMPLETED
                results[step.id] = step.result

                if progress_cb:
                    await progress_cb(f"Step {step.id}: done")
            except Exception as exc:
                step.status = WorkflowStatus.FAILED
                step.result = str(exc)
                workflow.status = WorkflowStatus.FAILED
                raise

        workflow.status = WorkflowStatus.COMPLETED
        return results

    async def get_status(self, workflow_id: str) -> WorkflowStatus:
        wf = self._workflows.get(workflow_id)
        return wf.status if wf else WorkflowStatus.PENDING

    async def cancel(self, workflow_id: str) -> None:
        wf = self._workflows.get(workflow_id)
        if wf:
            wf.status = WorkflowStatus.CANCELLED
