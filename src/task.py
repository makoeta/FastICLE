from typing import Annotated

from pydantic import BaseModel, Field


class DispatcherTask(BaseModel):
    order_number: Annotated[int, Field(description="Order of the task being executed.")]
    description: Annotated[str, Field(description="Description of the task.")]


class CasterTask(DispatcherTask):
    agent_id: Annotated[str, Field(description="Assigned Agent")]


class RuntimeTask(CasterTask):
    task_output: Annotated[str, Field()]


class DispatcherTaskList(BaseModel):
    task_list: list[DispatcherTask]


class CasterTaskList(BaseModel):
    task_list: list[CasterTask]


class RuntimeTaskList(BaseModel):
    task_list: Annotated[list[RuntimeTask], Field(default_factory=list)]
