import copy
import logging
from concurrent.futures import ThreadPoolExecutor

from agno.agent import Agent
from agno.models.base import Model
from agno.run.agent import RunOutput
from agno.team import Team
from agno.workflow import StepInput, StepOutput
from dotenv import load_dotenv
from pydantic import BaseModel

from icle.campus.models.expert_config import ExpertConfig
from icle.models.tasks import CasterTask, CasterTaskList, RuntimeTask, RuntimeTaskList
from icle.runtime.prompts import DEPENDENCY_CONTEXT_PREAMBLE

load_dotenv()
logger = logging.getLogger(__name__)


class Runtime(BaseModel):

    expert_save_dir: str
    model: Model

    def _build_batches(self, tasks: list[CasterTask]) -> list[list[CasterTask]]:
        known_ids = {t.task_id for t in tasks}
        for task in tasks:
            unknown = [dep for dep in task.depends_on if dep not in known_ids]
            if unknown:
                raise ValueError(
                    f"Task '{task.task_id}' depends on unknown task_id(s): {unknown}"
                )

        completed: set[str] = set()
        remaining = list(tasks)
        batches = []

        while remaining:
            ready = [
                t for t in remaining if all(dep in completed for dep in t.depends_on)
            ]
            if not ready:
                raise ValueError(
                    f"Dependency cycle detected among: {[t.task_id for t in remaining]}"
                )
            batches.append(ready)
            completed.update(t.task_id for t in ready)
            remaining = [t for t in remaining if t.task_id not in completed]

        return batches

    def runtime(self, step_input: StepInput) -> StepOutput:
        runtime_task_list: RuntimeTaskList = RuntimeTaskList()

        caster_task_list: CasterTaskList = step_input.get_last_step_content()

        prior_tasks_by_id: dict[str, RuntimeTask] = {}
        for batch in self._build_batches(caster_task_list.task_list):
            logger.info(
                "Running %d task(s) in parallel: %s",
                len(batch),
                [t.task_id for t in batch],
            )

            with ThreadPoolExecutor() as pool:
                futures = [
                    (
                        task,
                        pool.submit(
                            self._run_task,
                            task,
                            self._build_prior_context(task, prior_tasks_by_id),
                        ),
                    )
                    for task in batch
                ]

            for task, future in futures:
                runtime_task: RuntimeTask = future.result()
                logger.info(
                    "Task [%s] finished (%d chars, experts %s)",
                    runtime_task.task_id,
                    len(runtime_task.task_output),
                    runtime_task.agent_ids,
                )
                logger.debug(
                    "Task [%s] output:\n%s",
                    runtime_task.task_id,
                    runtime_task.task_output,
                )
                runtime_task_list.task_list.append(runtime_task)
                prior_tasks_by_id[runtime_task.task_id] = runtime_task

        return StepOutput(content=runtime_task_list)

    def _build_prior_context(
        self, task: CasterTask, prior_tasks_by_id: dict[str, RuntimeTask]
    ) -> str:
        deps = [
            prior_tasks_by_id[dep]
            for dep in task.depends_on
            if dep in prior_tasks_by_id
        ]
        if not deps:
            return ""

        dep_xmls = "\n".join(dep.to_context_xml() for dep in deps)
        return (
            f"{DEPENDENCY_CONTEXT_PREAMBLE}\n"
            f"<dependency_outputs>\n{dep_xmls}\n</dependency_outputs>"
        )

    def _run_task(self, caster_task: CasterTask, prior_context: str = ""):
        if not caster_task.agent_ids:
            raise ValueError(
                f"Task '{caster_task.task_id}' has no assigned agents — cannot execute."
            )

        path: str = self.expert_save_dir
        experts: list[Agent] = []

        for agent_id in caster_task.agent_ids:
            expert: Agent = ExpertConfig.from_yaml(
                path + "/" + agent_id + ".yaml"
            ).to_agent()
            experts.append(expert)

        team: Team = Team(members=experts, model=copy.deepcopy(self.model))

        prompt = (
            f"{prior_context}\n\n{caster_task.description}"
            if prior_context
            else caster_task.description
        )

        logger.debug("Task [%s] prompt:\n%s", caster_task.task_id, prompt)

        expert_out: RunOutput = team.run(prompt)

        if expert_out.content is None:
            raise RuntimeError(
                f"Task '{caster_task.task_id}' produced no output (team.run returned content=None)"
            )

        return RuntimeTask(**caster_task.model_dump(), task_output=expert_out.content)
