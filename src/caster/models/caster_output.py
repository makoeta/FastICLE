from typing import Annotated

from pydantic import BaseModel, Field


class TaskAssignment(BaseModel):
    agent_id: str = Field(description="The identifier of the agent.")
    task_id: int = Field(description="The integer ID of the task.")


class CasterOutput(BaseModel):
    assigned_tasks: list[TaskAssignment] = Field(
        description="A list of task assignments matching agents to their specific task IDs."
    )
