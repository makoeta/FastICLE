import copy
import logging
import os
from functools import wraps

from agno.agent import Agent
from agno.models.base import Model

from icle.campus.core import Campus
from icle.caster.prompts import build_casting_prompt, build_training_prompt

from icle.models.tasks import CasterTaskList

logger = logging.getLogger(__name__)


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
        # Phase-2 (assignment) is schema-constrained and carries NO tools:
        # strict structured output and native tool-calling cannot coexist in a
        # single model call, so training is done separately in phase-1.
        self.output_schema = CasterTaskList
        self.campus = campus
        self.global_task = global_task
        self.multi_expert_mode = multi_expert_mode

    def _expert_repr(self) -> str:
        experts = self.campus.get_experts()
        return "\n".join(
            f"<expert>\n\t<id>{e.name}</id>\n\t<description>{e.description}</description>\n</expert>"
            for e in experts
        )

    def _train_missing_experts(self, task_context: str) -> None:
        """Phase 1: a tool-enabled agent (no output_schema) that trains any
        experts the tasks require but the campus doesn't have yet. Tool calls
        actually execute here, unlike in the schema-constrained assignment
        pass where the model can only emit the CasterTaskList JSON."""

        def train_new_expert(
            expert_task: str, expert_name: str, short_description: str
        ) -> str:
            logger.info(f"Requesting new expert: {expert_name}")
            self.campus.train_new_expert(
                expert_name=expert_name,
                expert_task=expert_task,
                description=short_description,
            )
            return f"Trained expert '{expert_name}'."

        trainer = Agent(
            model=copy.deepcopy(self.model),
            system_message=build_training_prompt(
                global_task=self.global_task,
                available_experts=self._expert_repr(),
                multi_expert_mode=self.multi_expert_mode,
            ),
            tools=[train_new_expert],
        )
        trainer.run(input=task_context)

    def _ensure_assigned_experts_exist(self, task_list: CasterTaskList) -> None:
        """Safety net: if the assignment pass names an expert that was never
        trained, train it on demand so the Runtime never fails to load its
        config. Keeps a full benchmark run from aborting on a single stray id."""
        save_dir = self.campus.expert_save_dir
        trained: set[str] = set()

        for task in task_list.task_list:
            for agent_id in task.agent_ids:
                if agent_id in trained:
                    continue
                path = os.path.join(save_dir, agent_id + ".yaml")
                if not os.path.exists(path):
                    logger.warning(
                        f"Assigned expert '{agent_id}' was not trained; "
                        f"training it now as a fallback."
                    )
                    self.campus.train_new_expert(
                        expert_name=agent_id,
                        expert_task=task.description,
                        description=f"Fallback expert for: {task.description[:120]}",
                    )
                trained.add(agent_id)

    def update_system_message(self):
        self.system_message = build_casting_prompt(
            global_task=self.global_task,
            available_experts=self._expert_repr(),
            multi_expert_mode=self.multi_expert_mode,
        )

    @wraps(Agent.run)
    def run(self, *args, **kwargs):
        task_context = kwargs.get("input")
        if task_context is None and args:
            task_context = args[0]
        logger.debug("Caster input:\n%s", task_context)

        # Phase 1: ensure every needed expert exists (tool calls execute).
        self._train_missing_experts(str(task_context))

        # Phase 2: assign experts from the now-updated pool (schema output).
        self.update_system_message()
        output = super().run(*args, **kwargs)

        # Backstop against the model naming an untrained expert.
        if isinstance(output.content, CasterTaskList):
            self._ensure_assigned_experts_exist(output.content)
            for task in output.content.task_list:
                logger.info("Cast [%s] -> experts %s", task.task_id, task.agent_ids)

        return output
