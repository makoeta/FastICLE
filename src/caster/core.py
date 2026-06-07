from functools import wraps
from typing import Annotated

from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput
from pydantic import Field

from campus.core import Campus
from caster.models.caster_output import CasterOutput
from caster.prompts import casting_agent_system_prompt


class CasterAgent(Agent):

    campus: Campus
    global_task: str
    model: Model

    @wraps(Agent.__init__)
    def __init__(self, model: Model, campus: Campus, global_task: str, **kwargs):
        super().__init__(**kwargs, model=model, output_schema=CasterOutput)

        self.campus = campus
        self.global_task = global_task

        def train_new_expert(expert_task: str, expert_name: str) -> None:
            self.campus.train_new_expert(
                expert_name=expert_name, expert_task=expert_task
            )

        self.tools.append(train_new_expert)

    def update_system_message(self):
        experts = self.campus.get_experts()

        expert_repr = "\n".join(f"{e.name}: {e.task_description}" for e in experts)

        self.system_message = casting_agent_system_prompt.format(
            available_experts=expert_repr
        )

    @wraps(Agent.run)
    def run(self, *args, **kwargs) -> RunOutput:
        self.update_system_message()

        return super().run(*args, **kwargs)
