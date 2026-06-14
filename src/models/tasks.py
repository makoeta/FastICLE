from typing import Annotated
from xml.sax.saxutils import escape
from pydantic import BaseModel, Field


class DispatcherTask(BaseModel):
    order_number: Annotated[int, Field(description="Order of the task being executed.")]
    description: Annotated[str, Field(description="Description of the task.")]

    def __str__(self) -> str:
        lines = [f"[{self.__class__.__name__}]"]
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            lines.append(f"  {field_name}: {value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def to_xml(self) -> str:
        fields_xml = "\n\t".join(
            f"<{k}>{escape(str(getattr(self, k)))}</{k}>"
            for k in self.__class__.model_fields
        )
        return f"<task>\n\t{fields_xml}\n</task>"


class CasterTask(DispatcherTask):
    agent_ids: Annotated[
        list[str], Field(description="Assigned Agent", default_factory=list)
    ]


class RuntimeTask(CasterTask):
    task_output: Annotated[str, Field()]


class DispatcherTaskList(BaseModel):
    task_list: list[DispatcherTask]


class CasterTaskList(BaseModel):
    task_list: list[CasterTask]


class RuntimeTaskList(BaseModel):
    task_list: Annotated[list[RuntimeTask], Field(default_factory=list)]

    def to_xml(self) -> str:
        tasks_xml = "\n".join(task.to_xml() for task in self.task_list)
        return f"<tasks>\n{tasks_xml}\n</tasks>"
