from pydantic import Field
from typing import Annotated
from pydantic import BaseModel


from pydantic import BaseModel, Field


class TaskAssignment(BaseModel):
    agent_ids: list[str] = Field(
        description="A list of assigned agent IDs. (Use ['train_new_expert'] if no existing expert fits and a new one must be trained)."
    )
    task_id: int = Field(description="The integer ID of the task.")


class CasterOutput(BaseModel):
    assigned_tasks: list[TaskAssignment] = Field(
        description="A list of task assignments matching agents to their specific task IDs."
    )
