import logging
from functools import wraps
from typing import Annotated

from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput
from pydantic import Field

from campus.core import Campus
from caster.prompts import build_casting_prompt

from models.tasks import CasterTaskList

LOGGER = logging.getLogger(__name__)


class CasterAgent(Agent):

    campus: Campus
    global_task: str
    model: Model
    multi_expert_mode: bool

    @wraps(Agent.__init__)
    def __init__(
        self,
        model: Model,
        campus: Campus,
        global_task: str,
        multi_expert_mode: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.output_schema = CasterTaskList
        self.campus = campus
        self.global_task = global_task
        self.multi_expert_mode = multi_expert_mode

        def train_new_expert(
            expert_task: str, expert_name: str, short_description: str
        ) -> None:
            LOGGER.info(f"Requesting new expert: {expert_name}")
            self.campus.train_new_expert(
                expert_name=expert_name,
                expert_task=expert_task,
                description=short_description,
            )

        self.tools.append(train_new_expert)

    def update_system_message(self):
        experts = self.campus.get_experts()

        expert_repr = "\n".join(
            f"<expert>\n\t<id>{e.name}</id>\n\t<description>{e.description}</description>\n</expert>"
            for e in experts
        )

        self.system_message = build_casting_prompt(
            global_task=self.global_task,
            available_experts=expert_repr,
            multi_expert_mode=self.multi_expert_mode,
        )

    @wraps(Agent.run)
    def run(self, *args, **kwargs):
        self.update_system_message()

        return super().run(*args, **kwargs)
