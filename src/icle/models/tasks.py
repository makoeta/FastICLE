from typing import Annotated
from xml.sax.saxutils import escape
from pydantic import BaseModel, Field


class DispatcherTask(BaseModel):
    task_id: Annotated[
        str, Field(description="Unique identifier for this task (e.g. 'T1', 'T2').")
    ]
    description: Annotated[str, Field(description="Description of the task.")]
    depends_on: Annotated[
        list[str],
        Field(
            description="List of task_ids that must complete before this task starts. Empty list means no dependencies.",
            default_factory=list,
        ),
    ]

    def __str__(self) -> str:
        lines = [f"[{self.__class__.__name__}]"]
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            lines.append(f"  {field_name}: {value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()

    def to_xml(self) -> str:
        def _serialize(v) -> str:
            if isinstance(v, list):
                return escape(", ".join(str(i) for i in v))
            return escape(str(v))

        fields_xml = "\n\t".join(
            f"<{k}>{_serialize(getattr(self, k))}</{k}>"
            for k in self.__class__.model_fields
        )
        return f"<task>\n\t{fields_xml}\n</task>"


class CasterTask(DispatcherTask):
    agent_ids: Annotated[
        list[str], Field(description="Assigned Agent", default_factory=list)
    ]


class RuntimeTask(CasterTask):
    task_output: Annotated[str, Field()]

    def to_context_xml(self) -> str:
        """XML for injecting this task's result into a dependent task's
        prompt — only what the downstream task needs, without the
        agent/dependency bookkeeping that to_xml() carries."""
        return (
            "<task>\n"
            f"\t<task_id>{escape(self.task_id)}</task_id>\n"
            f"\t<description>{escape(self.description)}</description>\n"
            f"\t<task_output>{escape(self.task_output)}</task_output>\n"
            "</task>"
        )


class DispatcherTaskList(BaseModel):
    task_list: list[DispatcherTask]


class CasterTaskList(BaseModel):
    task_list: list[CasterTask]


class RuntimeTaskList(BaseModel):
    task_list: Annotated[list[RuntimeTask], Field(default_factory=list)]

    def to_xml(self) -> str:
        tasks_xml = "\n".join(task.to_xml() for task in self.task_list)
        return f"<tasks>\n{tasks_xml}\n</tasks>"
